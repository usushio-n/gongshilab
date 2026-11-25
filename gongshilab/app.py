
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import plotly.graph_objects as go

from dataclasses import dataclass
from torch.utils.data import Dataset


# =========================================================
# 1. 配置
# =========================================================

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CKPT_DIR = os.path.join(ROOT, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "poetry_latest.pt")

os.makedirs(CKPT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class Config:
    block_size: int = 32
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 1000
    eval_interval: int = 100


# =========================================================
# 2. 数据集与字符 Tokenizer
# =========================================================

class PoetryDataset:

    def __init__(self):
        files = glob.glob(os.path.join(DATA_DIR, "*.txt"))

        if not files:
            raise FileNotFoundError(
                "没有找到诗歌 TXT，请将文件放入 PoetryLab/data/"
            )

        self.text = ""

        for path in files:
            with open(path, "r", encoding="utf-8") as f:
                self.text += f.read() + "\n"

        self.chars = sorted(list(set(self.text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

        self.data = torch.tensor(
            [self.stoi[c] for c in self.text],
            dtype=torch.long
        )

        if len(self.data) < 300:
            raise ValueError("语料太短，建议至少准备几百个字符。")

        split_at = int(len(self.data) * 0.9)

        self.train_data = self.data[:split_at]
        self.val_data = self.data[split_at:]

        if len(self.val_data) <= Config.block_size:
            raise ValueError(
                "验证集太短，请增加诗歌语料，或减小 block_size。"
            )

    def encode(self, text):
        unknown = set(text) - set(self.stoi)

        if unknown:
            raise ValueError(
                "提示词包含训练词表中不存在的字符：" +
                "".join(sorted(unknown))
            )

        return [self.stoi[c] for c in text]

    def decode(self, tokens):
        return "".join(self.itos[int(i)] for i in tokens)

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data

        ix = torch.randint(
            len(data) - Config.block_size - 1,
            (TrainConfig.batch_size,)
        )

        x = torch.stack([
            data[i:i + Config.block_size] for i in ix
        ])

        y = torch.stack([
            data[i + 1:i + Config.block_size + 1] for i in ix
        ])

        return x.to(DEVICE), y.to(DEVICE)


# =========================================================
# 3. GPT 模型
# =========================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(
                config.block_size,
                config.block_size
            ))
        )

        self.last_attention = None

    def forward(self, x):

        B, T, C = x.shape

        q, k, v = self.qkv(x).chunk(3, dim=-1)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
        scores = scores / math.sqrt(self.head_dim)

        scores = scores.masked_fill(
            self.mask[:T, :T] == 0,
            float("-inf")
        )

        attention = F.softmax(scores, dim=-1)

        self.last_attention = attention.detach()

        attention = self.dropout(attention)

        out = attention @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.proj(out)


class FeedForward(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

        self.attention = CausalSelfAttention(config)
        self.ffwd = FeedForward(config)

    def forward(self, x):

        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))

        return x


class PoetryGPT(nn.Module):

    def __init__(self, vocab_size, config):
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            vocab_size, config.n_embd
        )

        self.position_embedding = nn.Embedding(
            config.block_size, config.n_embd
        )

        self.blocks = nn.Sequential(*[
            TransformerBlock(config)
            for _ in range(config.n_layer)
        ])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

    def forward(self, idx, targets=None, return_details=False):

        B, T = idx.shape

        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.shape[1]

        positions = torch.arange(T, device=idx.device)

        x = (
            self.token_embedding(idx)
            + self.position_embedding(positions)
        )

        hidden_states = []

        for block in self.blocks:
            x = block(x)
            hidden_states.append(x.detach())

        hidden = self.ln_f(x)
        logits = self.lm_head(hidden)

        loss = None

        if targets is not None:
            targets = targets[:, -T:]

            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

        if return_details:
            attentions = [
                block.attention.last_attention
                for block in self.blocks
            ]

            details = {
                "hidden": hidden.detach(),
                "attentions": attentions
            }

            return logits, loss, details

        return logits, loss


# =========================================================
# 4. 训练与模型保存
# =========================================================

model = None
dataset = None
optimizer = None
history = {"train": [], "val": []}
current_step = 0


@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(10):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()

    return results


def save_checkpoint():

    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "chars": dataset.chars,
        "config": {
        "block_size": Config.block_size,
        "n_embd": Config.n_embd,
        "n_head": Config.n_head,
        "n_layer": Config.n_layer,
        "dropout": Config.dropout
},
        "history": history,
        "step": current_step
    }, CKPT_PATH)


def train_model(mode, steps, progress=gr.Progress()):

    global model, dataset, optimizer, history, current_step

    dataset = PoetryDataset()

    if mode == "Continue":

        if not os.path.exists(CKPT_PATH):
            raise gr.Error("没有找到旧模型，请先选择 Scratch 训练。")

        checkpoint = torch.load(
            CKPT_PATH,
            map_location=DEVICE,
            weights_only=False
        )

        if checkpoint["chars"] != dataset.chars:
            raise gr.Error(
                "当前语料词表与旧模型不同。"
                "Continue 要求字符集合完全一致，请使用 Scratch。"
            )

        saved_config = checkpoint["config"]
        current_config = Config()

        for key, value in saved_config.items():
          if getattr(current_config, key) != value:
            raise gr.Error("模型配置不一致，无法 Continue。")

        model = PoetryGPT(
            len(dataset.chars),
            Config()
        ).to(DEVICE)

        model.load_state_dict(checkpoint["model"])

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TrainConfig.learning_rate
        )

        optimizer.load_state_dict(checkpoint["optimizer"])

        history = checkpoint["history"]
        current_step = checkpoint["step"]

    else:

        model = PoetryGPT(
            len(dataset.chars),
            Config()
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=TrainConfig.learning_rate
        )

        history = {"train": [], "val": []}
        current_step = 0

    model.train()

    steps = int(steps)

    for i in range(steps):

        x, y = dataset.get_batch("train")

        logits, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )

        optimizer.step()

        current_step += 1

        if current_step % 50 == 0 or i == steps - 1:

            losses = estimate_loss()

            history["train"].append(losses["train"])
            history["val"].append(losses["val"])

            save_checkpoint()

            progress(
                (i + 1) / steps,
                desc=f"Step {current_step} | "
                     f"Train {losses['train']:.4f} | "
                     f"Val {losses['val']:.4f}"
            )

            yield (
                f"训练中：Step {current_step}\n"
                f"Train Loss: {losses['train']:.4f}\n"
                f"Val Loss: {losses['val']:.4f}",
                make_loss_plot()
            )

    save_checkpoint()

    yield (
        f"训练完成！总 Step：{current_step}",
        make_loss_plot()
    )


# =========================================================
# 5. 诗歌生成
# =========================================================

@torch.no_grad()
def generate_poem(prompt, max_new_tokens, temperature, top_k):

    if model is None or dataset is None:
        raise gr.Error("请先训练模型。")

    model.eval()

    unknown = [c for c in prompt if c not in dataset.stoi]

    if unknown:
      prompt = "".join(
          c for c in prompt if c in dataset.stoi
      )

      if not prompt:
          raise gr.Error(
              "提示词全部是未知字符，请更换提示词。"
        )

      print("已过滤未知字符：", unknown)

    ids = dataset.encode(prompt)

    if not ids:
        ids = [0]

    idx = torch.tensor(
        [ids],
        dtype=torch.long,
        device=DEVICE
    )

    probabilities = None
    trace = []

    for _ in range(int(max_new_tokens)):

        context = idx[:, -Config.block_size:]

        logits, _, details = model(
            context,
            return_details=True
        )

        logits = logits[:, -1, :] / max(float(temperature), 0.01)

        if int(top_k) > 0:
            values, _ = torch.topk(
                logits, min(int(top_k), logits.size(-1))
            )

            logits[logits < values[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)

        next_id = torch.multinomial(probs, num_samples=1)

        probabilities = probs[0].cpu()

        trace.append({
            "attention": details["attentions"][-1][0, 0].cpu(),
            "hidden": details["hidden"][0, -1].cpu()
        })

        idx = torch.cat((idx, next_id), dim=1)

    model.train()

    poem = dataset.decode(idx[0].tolist())

    return (
        poem,
        make_probability_plot(probabilities),
        make_attention_plot(trace),
        make_hidden_plot(trace)
    )


# =========================================================
# 6. 可视化
# =========================================================

def make_loss_plot():

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        y=history["train"],
        mode="lines",
        name="Train Loss"
    ))

    fig.add_trace(go.Scatter(
        y=history["val"],
        mode="lines",
        name="Validation Loss"
    ))

    fig.update_layout(
        title="Training Loss",
        xaxis_title="Evaluation",
        yaxis_title="Loss"
    )

    return fig


def make_probability_plot(probs):

    if probs is None:
        return go.Figure()

    values, indices = torch.topk(
        probs, min(20, len(probs))
    )

    chars = [dataset.itos[int(i)] for i in indices]

    fig = go.Figure(go.Bar(
        x=chars,
        y=values.tolist()
    ))

    fig.update_layout(title="Next Character Probability")

    return fig


def make_attention_plot(trace):

    if not trace:
        return go.Figure()

    attention = trace[-1]["attention"].numpy()

    fig = go.Figure(go.Heatmap(z=attention))

    fig.update_layout(title="Last Layer Attention")

    return fig


def make_hidden_plot(trace):

    if not trace:
        return go.Figure()

    hidden = torch.stack([
        item["hidden"] for item in trace
    ]).numpy()

    fig = go.Figure(go.Heatmap(z=hidden.T))

    fig.update_layout(title="Hidden State During Generation")

    return fig


# =========================================================
# 7. Gradio 界面
# =========================================================

with gr.Blocks(title="龚诗生成器") as demo:

    gr.Markdown("# 龚诗生成器")
    gr.Markdown("字符级 GPT 中文现代诗生成实验室")

    with gr.Row():

        with gr.Column(scale=1):

            gr.Markdown("## 训练")

            mode = gr.Radio(
                ["Scratch", "Continue"],
                value="Scratch",
                label="训练模式"
            )

            steps = gr.Number(
                value=1000,
                label="训练步数",
                precision=0
            )

            train_button = gr.Button(
                "开始训练",
                variant="primary"
            )

            train_status = gr.Textbox(
                label="训练状态",
                lines=3
            )

            loss_plot = gr.Plot(label="Loss")

            gr.Markdown("## 生成")

            prompt = gr.Textbox(
                label="诗歌开头",
                value="春天",
                lines=2
            )

            max_tokens = gr.Slider(
                20, 300, value=100,
                step=10,
                label="生成字符数"
            )

            temperature = gr.Slider(
                0.1, 1.5, value=0.8,
                label="Temperature"
            )

            top_k = gr.Slider(
                0, 30, value=10,
                step=1,
                label="Top-K"
            )

            generate_button = gr.Button(
                "生成诗歌",
                variant="primary"
            )

            poem_output = gr.Textbox(
                label="生成结果",
                lines=10
            )

        with gr.Column(scale=1):

            gr.Markdown("## 模型观察")

            probability_plot = gr.Plot(
                label="下一字符概率"
            )

            attention_plot = gr.Plot(
                label="Attention"
            )

            hidden_plot = gr.Plot(
                label="Hidden State"
            )

    train_button.click(
        train_model,
        inputs=[mode, steps],
        outputs=[train_status, loss_plot]
    )

    generate_button.click(
        generate_poem,
        inputs=[
            prompt,
            max_tokens,
            temperature,
            top_k
        ],
        outputs=[
            poem_output,
            probability_plot,
            attention_plot,
            hidden_plot
        ]
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        debug=True
    )
