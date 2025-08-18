import discord
from discord.ext import commands
import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F
import os
from dotenv import load_dotenv

TOXICITY_THRESHOLD = 0.6
SPAN_THRESHOLD = 0.4
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
AVG_TOKENS = 90

def format_toxic_message(text, token_probs, offset_mapping, threshold=0.4):
    spans = []
    for (start, end), prob in zip(offset_mapping[0].tolist(), token_probs):
        if prob >= threshold and start != end:
            spans.append((start, end))

    # Merge overlapping/adjacent spans
    merged = []
    for s, e in sorted(spans):
        if not merged or s > merged[-1][1]:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)

    # Insert Markdown highlights
    result = []
    last = 0
    for s, e in merged:
        result.append(text[last:s])
        result.append(f"**{text[s:e]}**")
        last = e
    result.append(text[last:])

    return "".join(result)

# Load model
class ToxicityModel(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)

        # Heads
        self.seq_head = nn.Linear(hidden, 1)   # Sequence classification
        self.tok_head = nn.Linear(hidden, 2)   # Token-level classification

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        last_hidden = self.dropout(out.last_hidden_state)  # [B, T, H]
        cls_pooled = self.dropout(last_hidden[:, 0])      # CLS pooling

        seq_logits = self.seq_head(cls_pooled)            # [B, 1]
        tok_logits = self.tok_head(last_hidden)           # [B, T, 2]
        return seq_logits, tok_logits

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ToxicityModel(model_name="microsoft/deberta-v3-base")
checkpoint_path = "models/toxicity_span_epoch5.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
model.to(device)

intents = discord.Intents.default()
intents.message_content = True  # Needed to read messages
bot = commands.Bot(command_prefix="!", intents=intents)

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def predict_toxicity(text):
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=AVG_TOKENS,  # must match training
        return_offsets_mapping=True,
        return_tensors="pt"
    )
    offset_mapping = inputs.pop("offset_mapping")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Run model
    with torch.no_grad():
        seq_logits, tok_logits = model(**inputs)

        # Sequence-level probability
        toxicity_prob = torch.sigmoid(seq_logits).squeeze().item()

        # Token-level probabilities (take class=1 for toxic)
        token_probs = F.softmax(tok_logits, dim=-1)[..., 1].squeeze(0).tolist()

    # Decide overall toxic/non-toxic
    is_toxic = toxicity_prob >= TOXICITY_THRESHOLD

    return is_toxic, token_probs, offset_mapping

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return  # ignore self

    is_toxic, token_probs, offset_mapping = predict_toxicity(message.content)

    if is_toxic:
        try:
            print("Toxic message")
            # delete the original toxic message
            await message.delete()

            # highlight spans in bold using offsets
            highlighted = format_toxic_message(message.content, token_probs, offset_mapping)

            # DM the user
            dm_text = f"Message flagged due to toxicity:\n{highlighted}"
            await message.author.send(dm_text)

        except Exception as e:
            print(f"Error handling toxic message: {e}")

    # Important: process commands if any
    await bot.process_commands(message)

# Run your bot
bot.run(BOT_TOKEN)
