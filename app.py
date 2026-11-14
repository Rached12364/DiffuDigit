"""Application DiffuDigit.   python app.py   puis ouvrir http://127.0.0.1:7860"""
import io
import numpy as np
import torch
import gradio as gr
from PIL import Image
from model import UNet, generate

device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet().to(device)
model.load_state_dict(torch.load("diffudigit.pt", map_location=device))
model.eval()
STEPS = (399, 300, 200, 100, 50, 0)


def to_pil(x, size=224):
    a = ((x.squeeze().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a).resize((size, size), Image.NEAREST)


def run(digit, w, seed):
    final, frames = generate(model, int(digit), n=1, w=float(w), keep=STEPS, device=device, seed=int(seed))
    etapes = [(to_pil(frames[s][0], 160), f"Étape {s}") for s in STEPS]
    # animation GIF du débruitage
    _, all_frames = generate(model, int(digit), n=1, w=float(w), keep=tuple(range(0, 400, 10)), device=device, seed=int(seed))
    gif = [to_pil(all_frames[s][0]) for s in sorted(all_frames, reverse=True)]
    gif += [gif[-1]] * 8
    path = "debruitage.gif"
    gif[0].save(path, save_all=True, append_images=gif[1:], duration=60, loop=0)
    return to_pil(final[0], 280), etapes, path


with gr.Blocks(title="DiffuDigit") as demo:
    gr.Markdown("# DiffuDigit\nChoisissez un chiffre : le modèle part d’un bruit aléatoire et le retire, étape par étape.")
    with gr.Row():
        with gr.Column(scale=1):
            digit = gr.Radio([str(i) for i in range(10)], value="7", label="Chiffre à générer")
            w = gr.Slider(0, 6, value=2, step=0.5, label="Guidance (CFG)")
            seed = gr.Number(value=42, precision=0, label="Graine aléatoire")
            btn = gr.Button("Générer", variant="primary")
        with gr.Column(scale=1):
            out = gr.Image(label="Image générée", type="pil")
            anim = gr.Image(label="Débruitage animé", type="filepath")
    steps = gr.Gallery(label="Du bruit à l’image", columns=6, height=220)
    btn.click(run, [digit, w, seed], [out, steps, anim])

if __name__ == "__main__":
    demo.launch()
