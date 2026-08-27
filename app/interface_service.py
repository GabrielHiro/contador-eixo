#!/usr/bin/env python3
"""Interface web local para operar o contador de eixos como um serviço LPR."""

from __future__ import annotations

import cgi
import html
import json
import os
import sqlite3
import subprocess
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "ui_jobs"
DB_PATH = JOBS / "history.sqlite3"
BACKEND = os.environ.get("CONTADOR_API", "http://127.0.0.1:8080/api/count")
PORT = int(os.environ.get("CONTADOR_UI_PORT", "8090"))


def database() -> sqlite3.Connection:
    JOBS.mkdir(exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, original_name TEXT NOT NULL, video_name TEXT NOT NULL,
        processed_video_name TEXT,
        created_at TEXT NOT NULL, state TEXT NOT NULL, vehicles INTEGER,
        axles INTEGER, frames INTEGER, processing_seconds REAL, cpu_percent REAL,
        config_json TEXT NOT NULL, error TEXT)""")
    columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
    if "processed_video_name" not in columns:
        db.execute("ALTER TABLE jobs ADD COLUMN processed_video_name TEXT")
    db.execute("""CREATE TABLE IF NOT EXISTS camera_configs (
        name TEXT PRIMARY KEY, updated_at TEXT NOT NULL, config_json TEXT NOT NULL)""")
    db.commit()
    return db


def cpu_sample() -> tuple[int, int]:
    with open("/proc/stat", encoding="ascii") as stream:
        fields = stream.readline().split()
    values = [int(value) for value in fields[1:8]]
    return sum(values), values[0] + values[1] + values[2]


def cpu_usage(before: tuple[int, int], after: tuple[int, int]) -> float:
    total = after[0] - before[0]
    busy = after[1] - before[1]
    return max(0.0, min(100.0, busy * 100.0 / total)) if total else 0.0


def browser_video(source: Path, target: Path) -> Path:
    """Converte mp4v para H.264, codec suportado pelo elemento video do navegador."""
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source), "-c:v", "libx264",
             "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(target)],
            check=True,
            timeout=600,
        )
        return target if target.is_file() else source
    except (OSError, subprocess.SubprocessError):
        return source


def esc(value: object) -> str:
    return html.escape(str(value if value is not None else "-"))


def nav(active: str) -> str:
    return f'''<nav><a class="{"active" if active == "process" else ""}" href="/">Vídeo</a>
    <a class="{"active" if active == "image" else ""}" href="/image">Imagem</a>
    <a class="{"active" if active == "history" else ""}" href="/history">Histórico</a></nav>'''


def layout(content: str, active: str = "process", notice: str = "") -> bytes:
    return f'''<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Radar | Eixos</title>
<style>
:root{{--ink:#eaf0f2;--muted:#91a4a8;--panel:#182126;--line:#304044;--yellow:#f7c948;--green:#55d68a;--red:#ff8c8c}}
*{{box-sizing:border-box}}body{{margin:0;background:#0e1417;color:var(--ink);font:15px system-ui,sans-serif}}
main{{max-width:1180px;margin:auto;padding:30px 20px 60px}}header{{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:26px}}
h1{{font-size:clamp(2.2rem,5vw,4.5rem);line-height:.9;margin:5px 0 0;color:var(--yellow);letter-spacing:-.02em}}h2{{margin:0 0 18px;font-size:1.1rem}}
.eyebrow{{color:var(--muted);text-transform:uppercase;letter-spacing:.16em;font-size:.72rem}}nav{{display:flex;gap:8px;margin-bottom:25px;border-bottom:1px solid var(--line)}}nav a{{color:var(--muted);text-decoration:none;padding:12px 16px 10px;border-bottom:2px solid transparent}}nav a.active,nav a:hover{{color:var(--ink);border-color:var(--yellow)}}
.grid{{display:grid;grid-template-columns:minmax(280px,400px) 1fr;gap:22px}}.panel{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:22px}}.wide{{grid-column:1/-1}}
label{{display:block;color:#b6c4c6;font-size:.82rem;margin:14px 0 0}}input,select{{width:100%;margin-top:6px;padding:10px;background:#0d1215;border:1px solid #435257;border-radius:5px;color:#fff;font-size:.95rem}}input[type=file]{{padding:8px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.line-tools{{display:flex;gap:8px;align-items:end}}.line-tools .row{{flex:1}}.section-title{{font-size:.76rem;text-transform:uppercase;letter-spacing:.1em;color:var(--yellow);margin:22px 0 -4px}}button{{width:100%;padding:13px;margin-top:20px;border:0;border-radius:5px;background:var(--yellow);font-weight:800;cursor:pointer}}button:hover{{background:#ffe07b}}button.secondary{{background:#2d4145;color:var(--ink);margin-top:10px}}button.danger{{width:auto;margin:0;padding:8px 12px;background:#522d2d;color:#ffd0d0}}
.hint,.meta{{color:var(--muted);font-size:.82rem;line-height:1.5}}.video{{width:100%;max-height:58vh;background:#070a0b;border-radius:5px}}.empty{{min-height:280px;display:grid;place-items:center;color:#718286;text-align:center}}
.image-result{{display:block;width:100%;height:auto;max-height:70vh;object-fit:contain;background:#070a0b;border-radius:5px}}
.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:16px}}.metric{{background:#20342d;border:1px solid #3d7258;border-radius:6px;padding:13px}}.metric span{{display:block;color:#a7c2b2;font-size:.75rem}}.metric strong{{display:block;font-size:1.55rem;margin-top:5px}}.error{{color:var(--red)}}
.toolbar{{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-bottom:15px}}.toolbar input{{max-width:300px;margin:0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px 8px;text-align:left;border-bottom:1px solid var(--line);vertical-align:middle}}th{{color:var(--muted);font-size:.75rem;text-transform:uppercase}}td strong{{color:var(--yellow);font-size:1.2rem}}td a{{color:var(--ink);text-decoration:none}}.status{{color:var(--green)}}pre{{white-space:pre-wrap;color:var(--muted)}}
.modal{{display:none;position:fixed;inset:0;background:#000b;z-index:5;padding:4vh 4vw}}.modal.open{{display:grid;place-items:center}}.modal-box{{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:18px;width:min(1000px,96vw);max-height:92vh;overflow:auto}}.draw-area{{position:relative;background:#080c0e;text-align:center;margin-top:12px}}.draw-area video{{max-height:70vh;display:block;margin:auto}}.draw-area canvas{{position:absolute;inset:0;width:100%;height:100%;cursor:crosshair}}.modal-actions{{display:flex;gap:8px;justify-content:flex-end}}.modal-actions button{{width:auto;padding:10px 16px;margin-top:14px}}
@media(max-width:780px){{main{{padding:22px 14px}}header{{display:block}}.grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:1fr 1fr}}.toolbar{{display:block}}.toolbar input{{max-width:none;margin-top:10px}}table{{font-size:.82rem}}th:nth-child(3),td:nth-child(3){{display:none}}}}
</style></head><body><main><header><div><div class="eyebrow">Radar / LPR</div><h1>Contagem de eixos</h1></div><div class="hint">um veículo por vídeo</div></header>{nav(active)}{notice}{content}</main></body></html>'''.encode("utf-8")


def process_page(result: dict | None = None, video_url: str = "", error: str = "") -> bytes:
    db = database()
    profiles = db.execute("SELECT name, config_json FROM camera_configs ORDER BY name").fetchall()
    db.close()
    profile_data = json.dumps({row["name"]: json.loads(row["config_json"]) for row in profiles}).replace("</", "<\\/")
    profile_options = "".join(f'<option value="{esc(row["name"])}">{esc(row["name"])}</option>' for row in profiles)
    preview = f'<video class="video" controls preload="metadata" src="{esc(video_url)}"></video>' if video_url else '<div class="empty">O vídeo processado aparecerá aqui.</div>'
    result_html = ""
    if result:
        result_html = f'''<div class="metrics"><div class="metric"><span>Veículos</span><strong>{esc(result.get("vehicles"))}</strong></div>
        <div class="metric"><span>Eixos</span><strong>{esc(result.get("axles"))}</strong></div>
        <div class="metric"><span>Tempo</span><strong>{esc(result.get("processing_seconds"))} s</strong></div>
        <div class="metric"><span>CPU média</span><strong>{esc(result.get("cpu_percent"))}%</strong></div></div>
        <p class="{"error" if error else "status"}">{esc(error or "Processamento concluído")}</p>'''
    content = f'''<div class="grid"><section class="panel"><h2>Novo processamento</h2><form id="process-form" method="post" enctype="multipart/form-data">
    <label>Vídeo do radar<input type="file" name="video" accept="video/*" required></label>
    <div class="section-title">Configuração da câmera</div><div class="row"><label>Perfil salvo<select id="profile-select"><option value="">Configuração manual</option>{profile_options}</select></label><label>Nome do novo perfil<input id="profile-name" placeholder="Ex.: Radar entrada norte"></label></div>
    <div class="row"><button type="button" class="secondary" onclick="saveProfile()">Salvar configuração da câmera</button><button type="button" class="secondary" onclick="openLineEditor()">Desenhar área no vídeo</button></div>
    <div class="section-title">Linha de contagem</div><div class="line-tools"><div class="row">
    <label>x1<input name="line_x1" type="number" step="0.01" value="180"></label><label>y1<input name="line_y1" type="number" step="0.01" value="160"></label>
    <label>x2<input name="line_x2" type="number" step="0.01" value="1100"></label><label>y2<input name="line_y2" type="number" step="0.01" value="620"></label></div></div>
    <div class="section-title">Detector de veículos</div><div class="row"><label>Confiança<input name="conf" type="number" min="0" max="1" step="0.01" value="0.10"></label><label>NMS<input name="nms" type="number" min="0" max="1" step="0.01" value="0.45"></label></div>
    <label>Modelo de veículos<input name="model_path" value="models/yolo26s.onnx"></label>
    <label>Classes YOLO26s (ordem dos IDs)<input name="vehicle_class_names" value="person,bicycle,car,motorcycle,airplane,bus,train,truck"><span class="hint">Modelo oficial COCO; o filtro abaixo mantém apenas classes rodoviárias.</span></label>
    <label>Filtro de classes (IDs separados por vírgula)<input name="vehicle_class_filter" value="7"><span class="hint">Os vídeos do radar devem conter somente caminhões. COCO: 7 = caminhão.</span></label>
    <div class="row"><label>ID leve<input name="light_vehicle_class_id" type="number" step="1" value="2"></label><label>ID moto<input name="motorcycle_class_id" type="number" step="1" value="3"></label></div>
    <div class="row"><label>ID caminhão<input name="truck_class_id" type="number" step="1" value="7"></label><label>Eixos caminhão (fixo)<input name="truck_axles_override" type="number" min="0" max="10" step="1" value="7"><span class="hint">Use 0 somente quando houver detector real de rodas.</span></label></div>
    <div class="section-title">Detector de eixos</div><div class="row"><label>Confiança<input name="axle_conf" type="number" min="0" max="1" step="0.01" value="0.35"></label><label>NMS<input name="axle_nms" type="number" min="0" max="1" step="0.01" value="0.45"></label></div>
    <label>Modelo de eixos<input name="axle_model_path" value="models/axles.onnx"></label>
    <div class="row"><label>Imagem veículo<input name="imgsz" type="number" min="64" step="1" value="640"></label><label>Imagem eixos<input name="axle_imgsz" type="number" min="64" step="1" value="224"></label></div>
    <label>Margem do recorte<input name="axle_crop_margin" type="number" min="0" max="1" step="0.01" value="0.15"></label>
    <label>Distância para agrupar rodas<input name="axle_group_distance" type="number" min="0.01" max="1" step="0.01" value="0.12"><span class="hint">Proporção do comprimento do veículo; cada grupo longitudinal representa um eixo.</span></label>
    <div class="row"><label>Eixos veículo leve<input name="light_vehicle_axles" type="number" min="2" max="6" step="1" value="2"></label><label>Eixos moto<input name="motorcycle_axles" type="number" min="2" max="6" step="1" value="2"></label></div>
    <button type="submit">Processar vídeo</button></form><p class="hint">As configurações são registradas junto ao resultado e só valem para este job.</p></section>
    <section class="panel"><h2>Resultado</h2>{preview}{result_html}</section></div>
    <div id="line-modal" class="modal" aria-hidden="true"><div class="modal-box"><div class="toolbar"><h2>Desenhar linha na imagem</h2><button type="button" class="danger" onclick="closeLineEditor()">Fechar</button></div><p class="hint">Clique no primeiro ponto e depois no segundo. A linha será aplicada automaticamente aos campos.</p><div class="draw-area"><video id="draw-video" controls muted></video><canvas id="draw-canvas"></canvas></div><div class="modal-actions"><button type="button" class="secondary" onclick="clearLine()">Limpar pontos</button><button type="button" onclick="closeLineEditor()">Aplicar linha</button></div></div></div>
    <script>const profiles = {profile_data};
    const field = name => document.querySelector('[name="' + name + '"]');
    const fileInput = document.querySelector('[name="video"]');
    const canvas = document.getElementById('draw-canvas'); const drawVideo = document.getElementById('draw-video'); let points = [];
    function resizeCanvas() {{ if (!drawVideo.videoWidth) return; canvas.width = drawVideo.videoWidth; canvas.height = drawVideo.videoHeight; draw(); }}
    function draw() {{ const ctx = canvas.getContext('2d'); ctx.clearRect(0,0,canvas.width,canvas.height); if (!points.length) return; ctx.lineWidth=4; ctx.strokeStyle='#f7c948'; ctx.fillStyle='#55d68a'; ctx.beginPath(); ctx.moveTo(points[0][0],points[0][1]); if(points[1]) ctx.lineTo(points[1][0],points[1][1]); ctx.stroke(); points.forEach(p => {{ ctx.beginPath(); ctx.arc(p[0],p[1],8,0,Math.PI*2); ctx.fill(); }}); }}
    function openLineEditor() {{ if (!fileInput.files[0]) {{ alert('Selecione um vídeo primeiro.'); return; }} drawVideo.src=URL.createObjectURL(fileInput.files[0]); document.getElementById('line-modal').classList.add('open'); drawVideo.addEventListener('loadedmetadata', resizeCanvas, {{once:true}}); }}
    function closeLineEditor() {{ document.getElementById('line-modal').classList.remove('open'); drawVideo.pause(); }}
    function clearLine() {{ points=[]; draw(); }}
    canvas.addEventListener('click', event => {{ const box=canvas.getBoundingClientRect(); const x=(event.clientX-box.left)*canvas.width/box.width; const y=(event.clientY-box.top)*canvas.height/box.height; if(points.length>=2) points=[]; points.push([x,y]); if(points.length===2) {{ field('line_x1').value=points[0][0].toFixed(2); field('line_y1').value=points[0][1].toFixed(2); field('line_x2').value=points[1][0].toFixed(2); field('line_y2').value=points[1][1].toFixed(2); }} draw(); }});
    window.addEventListener('resize', resizeCanvas); document.getElementById('profile-select').addEventListener('change', event => {{ const data=profiles[event.target.value]; if(!data) return; Object.keys(data).forEach(key => {{ if(field(key)) field(key).value=data[key]; }}); }});
    async function saveProfile() {{ const name=document.getElementById('profile-name').value.trim(); if(!name) {{ alert('Informe um nome para a câmera.'); return; }} const data={{}}; document.querySelectorAll('#process-form [name]').forEach(node => {{ if(node.name!=='video') data[node.name]=node.value; }}); const response=await fetch('/config/save', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{name,config:data}})}}); const result=await response.json(); alert(result.message || result.error); if(response.ok) location.reload(); }}
    </script>'''
    return layout(content)


def history_page(query: str = "") -> bytes:
    db = database()
    rows = db.execute("SELECT * FROM jobs WHERE original_name LIKE ? ORDER BY created_at DESC", (f"%{query}%",)).fetchall()
    db.close()
    body = "".join(f'''<tr><td><a href="/job/{esc(row["id"])}">{esc(row["original_name"])}</a><br><span class="meta">{esc(row["created_at"])}</span></td>
    <td class="status">{esc(row["state"])}</td><td><strong>{esc(row["axles"])}</strong> eixos<br><span class="meta">{esc(row["vehicles"])} veículo(s)</span></td>
    <td>{esc(row["processing_seconds"])} s<br><span class="meta">CPU {esc(row["cpu_percent"])}%</span></td></tr>''' for row in rows)
    if not body:
        body = '<tr><td colspan="4" class="meta">Nenhum processamento salvo.</td></tr>'
    content = f'''<section class="panel wide"><div class="toolbar"><div><h2>Processamentos salvos</h2><span class="meta">{len(rows)} registro(s)</span></div><form><input name="q" value="{esc(query)}" placeholder="Filtrar por nome"></form></div>
    <table><thead><tr><th>Vídeo</th><th>Estado</th><th>Resultado</th><th>Desempenho</th></tr></thead><tbody>{body}</tbody></table></section>'''
    return layout(content, "history")


def image_page(result: dict | None = None, image_url: str = "", error: str = "") -> bytes:
    preview = f'<img class="image-result" src="{esc(image_url)}">' if image_url else '<div class="empty">A imagem anotada aparecerá aqui.</div>'
    metrics = ""
    if result:
        metrics = f'''<div class="metrics"><div class="metric"><span>Veículo</span><strong>{esc(result.get("vehicle", "-"))}</strong></div>
        <div class="metric"><span>Eixos</span><strong>{esc(result.get("axles", "-"))}</strong></div><div class="metric"><span>Confiança</span><strong>{esc(result.get("confidence", "-"))}</strong></div></div>
        <p class="{"error" if error else "status"}">{esc(error or "Imagem processada")}</p>'''
    content = f'''<div class="grid"><section class="panel"><h2>Contar eixos em imagem</h2><form method="post" enctype="multipart/form-data" action="/image">
    <label>Imagem do veículo<input type="file" name="image" accept="image/*" required></label>
    <label>Modelo de veículos<input name="vehicle_model" value="models/yolo26s.onnx"></label>
    <label>Modelo de eixos<input name="axle_model" value="models/axles.onnx"></label>
    <div class="row"><label>Confiança veículo<input name="vehicle_conf" type="number" min="0" max="1" step="0.01" value="0.10"></label><label>Confiança eixo<input name="axle_conf" type="number" min="0" max="1" step="0.01" value="0.10"></label></div>
    <button type="submit">Processar imagem</button></form><p class="hint">A imagem é processada sem tracker: uma caixa principal do veículo e as detecções de eixos são desenhadas.</p></section>
    <section class="panel"><h2>Resultado</h2>{preview}{metrics}</section></div>'''
    return layout(content, "image")


def job_page(job_id: str) -> bytes:
    db = database(); row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone(); db.close()
    if row is None:
        return layout('<section class="panel"><p>Processamento não encontrado.</p></section>', "history")
    config = json.loads(row["config_json"])
    content = f'''<section class="panel"><div class="toolbar"><h2>{esc(row["original_name"])}</h2><form method="post" action="/delete/{esc(job_id)}"><button class="danger" type="submit">Excluir registro</button></form></div>
    <p class="meta">{esc(row["created_at"])} · estado: {esc(row["state"])}</p><video class="video" controls src="/media/{esc(row["processed_video_name"] or row["video_name"])}"></video>
    <div class="metrics"><div class="metric"><span>Veículos</span><strong>{esc(row["vehicles"])}</strong></div><div class="metric"><span>Eixos</span><strong>{esc(row["axles"])}</strong></div><div class="metric"><span>Tempo</span><strong>{esc(row["processing_seconds"])} s</strong></div><div class="metric"><span>CPU média</span><strong>{esc(row["cpu_percent"])}%</strong></div></div>
    <details style="margin-top:20px"><summary>Configuração usada</summary><pre>{esc(json.dumps(config, indent=2, ensure_ascii=False))}</pre></details></section>'''
    return layout(content, "history")


class Handler(BaseHTTPRequestHandler):
    def send_page(self, body: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self.send_page(process_page())
            return
        if path == "/image":
            self.send_page(image_page())
            return
        if path == "/history":
            query = parse_qs(parsed.query).get("q", [""])[0]
            self.send_page(history_page(query))
            return
        if path.startswith("/job/"):
            self.send_page(job_page(path.removeprefix("/job/")))
            return
        if path.startswith("/media/"):
            target = (JOBS / Path(path.removeprefix("/media/"))).resolve()
            if target.parent != JOBS.resolve() or not target.is_file():
                self.send_error(404)
                return
            file_size = target.stat().st_size
            range_header = self.headers.get("Range")
            start = 0
            end = file_size - 1
            if range_header and range_header.startswith("bytes="):
                requested = range_header[6:].split("-", 1)
                start = int(requested[0] or 0)
                end = int(requested[1]) if len(requested) > 1 and requested[1] else end
                end = min(end, file_size - 1)
                if start > end or start >= file_size:
                    self.send_error(416)
                    return
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            else:
                self.send_response(200)
            content_length = end - start + 1
            self.send_header("Accept-Ranges", "bytes")
            content_type = "image/jpeg" if target.suffix.lower() in (".jpg", ".jpeg") else "image/png" if target.suffix.lower() == ".png" else "video/mp4"
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(content_length))
            self.end_headers()
            with target.open("rb") as stream:
                stream.seek(start)
                remaining = content_length
                while remaining and (chunk := stream.read(min(1024 * 1024, remaining))):
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return
        self.send_error(404)

    def do_HEAD(self) -> None:
        path = urlparse(self.path).path
        if path.startswith("/media/"):
            target = (JOBS / Path(path.removeprefix("/media/"))).resolve()
            if target.parent != JOBS.resolve() or not target.is_file():
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(target.stat().st_size))
            self.end_headers()
            return
        self.send_error(404)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/image":
            form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
            field = form["image"] if "image" in form else None
            if field is None or not getattr(field, "filename", ""):
                self.send_page(image_page(error="Selecione uma imagem."), 400)
                return
            JOBS.mkdir(exist_ok=True)
            job_id = uuid.uuid4().hex
            image_name = f"{job_id}.jpg"
            image_path = JOBS / image_name
            output_name = f"{job_id}_annotated.jpg"
            output_path = JOBS / output_name
            with image_path.open("wb") as stream:
                while chunk := field.file.read(1024 * 1024):
                    stream.write(chunk)
            try:
                command = [str(ROOT / "venv/bin/python"), str(ROOT / "app/image_inference.py"), str(image_path), str(output_path), "--vehicle-model", form.getfirst("vehicle_model") or "models/yolo26s.onnx", "--axle-model", form.getfirst("axle_model") or "models/axles.onnx", "--vehicle-conf", form.getfirst("vehicle_conf") or "0.10", "--axle-conf", form.getfirst("axle_conf") or "0.10"]
                completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                self.send_page(image_page(result, f"/media/{output_name}"))
            except Exception as error:
                self.send_page(image_page(error=f"Falha no processamento: {error}"), 502)
            return
        if path == "/config/save":
            length = int(self.headers.get("Content-Length", "0"))
            try:
                data = json.loads(self.rfile.read(length).decode("utf-8"))
                name = str(data.get("name", "")).strip()
                config = data.get("config")
                if not name or not isinstance(config, dict):
                    raise ValueError("nome ou configuração inválida")
                db = database()
                db.execute("INSERT OR REPLACE INTO camera_configs VALUES (?, datetime('now','localtime'), ?)", (name, json.dumps(config)))
                db.commit(); db.close()
                response = json.dumps({"message": f"Configuração '{name}' salva."}).encode()
                self.send_response(200)
            except (ValueError, json.JSONDecodeError) as error:
                response = json.dumps({"error": str(error)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers(); self.wfile.write(response)
            return
        if path.startswith("/delete/"):
            job_id = path.removeprefix("/delete/")
            db = database()
            row = db.execute("SELECT video_name, processed_video_name FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row:
                (JOBS / row["video_name"]).unlink(missing_ok=True)
                if row["processed_video_name"]:
                    (JOBS / row["processed_video_name"]).unlink(missing_ok=True)
                db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
                db.commit()
            db.close()
            self.send_response(303)
            self.send_header("Location", "/history")
            self.end_headers()
            return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={"REQUEST_METHOD": "POST"})
        field = form["video"] if "video" in form else None
        if field is None or not getattr(field, "filename", ""):
            self.send_page(process_page(error="Selecione um vídeo."), 400)
            return
        JOBS.mkdir(exist_ok=True)
        job_id = uuid.uuid4().hex
        original = Path(field.filename).name
        video_name = f"{job_id}{Path(original).suffix.lower() or '.mp4'}"
        processed_video_name = f"{job_id}_processed.mp4"
        video_path = JOBS / video_name
        processed_path = JOBS / processed_video_name
        with video_path.open("wb") as stream:
            while chunk := field.file.read(1024 * 1024):
                stream.write(chunk)
        try:
            def number(name: str, default: str) -> float:
                return float(form.getfirst(name) or default)

            payload = {
                "video": str(video_path), "model_path": form.getfirst("model_path") or "models/yolo26s.onnx",
                "output_path": str(processed_path),
                "conf": number("conf", "0.10"), "nms": number("nms", "0.45"), "imgsz": int(number("imgsz", "640")),
                "line_x1": number("line_x1", "180"), "line_y1": number("line_y1", "160"),
                "line_x2": number("line_x2", "1100"), "line_y2": number("line_y2", "620"),
                "axle_model_path": form.getfirst("axle_model_path") or "models/axles.onnx",
                "axle_conf": number("axle_conf", "0.35"), "axle_nms": number("axle_nms", "0.45"),
                "axle_imgsz": int(number("axle_imgsz", "224")), "axle_crop_margin": number("axle_crop_margin", "0.15"),
                "vehicle_class_names": form.getfirst("vehicle_class_names") or "vehicle",
                "vehicle_class_filter": form.getfirst("vehicle_class_filter") or "7",
                "light_vehicle_class_id": int(number("light_vehicle_class_id", "2")),
                "motorcycle_class_id": int(number("motorcycle_class_id", "3")),
                "truck_class_id": int(number("truck_class_id", "7")),
                "truck_axles_override": int(number("truck_axles_override", "7")),
                "light_vehicle_axles": int(number("light_vehicle_axles", "2")),
                "motorcycle_axles": int(number("motorcycle_axles", "2")),
                "wheel_class_id": int(number("wheel_class_id", "0")),
                "axle_group_distance": number("axle_group_distance", "0.12"),
            }
            started = time.monotonic()
            before = cpu_sample()
            request = urllib.request.Request(BACKEND, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(request, timeout=7200) as response:
                result = json.loads(response.read().decode())
            result["processing_seconds"] = round(time.monotonic() - started, 2)
            result["cpu_percent"] = round(cpu_usage(before, cpu_sample()), 1)
            browser_name = f"{job_id}_browser.mp4"
            display_path = browser_video(processed_path, JOBS / browser_name)
            display_name = display_path.name
            db = database()
            db.execute("INSERT INTO jobs (id, original_name, video_name, created_at, state, vehicles, axles, frames, processing_seconds, cpu_percent, config_json, error, processed_video_name) VALUES (?, ?, ?, datetime('now','localtime'), ?, ?, ?, ?, ?, ?, ?, ?, ?)", (job_id, original, video_name, result.get("state", "finished"), result.get("vehicles"), result.get("axles"), result.get("frames"), result["processing_seconds"], result["cpu_percent"], json.dumps(payload), result.get("error"), display_name))
            db.commit(); db.close()
            self.send_page(process_page(result, f"/media/{display_name}"))
        except Exception as error:
            error_text = f"Falha no processamento: {error}"
            db = database()
            db.execute("INSERT INTO jobs (id, original_name, video_name, created_at, state, vehicles, axles, frames, processing_seconds, cpu_percent, config_json, error, processed_video_name) VALUES (?, ?, ?, datetime('now','localtime'), ?, ?, ?, ?, ?, ?, ?, ?, ?)", (job_id, original, video_name, "error", None, None, None, None, None, json.dumps(payload if 'payload' in locals() else {}), error_text, processed_video_name))
            db.commit(); db.close()
            self.send_page(process_page({"vehicles": "-", "axles": "-", "processing_seconds": "-", "cpu_percent": "-"}, f"/media/{video_name}", error_text), 502)


if __name__ == "__main__":
    database().close()
    print(f"Interface em http://127.0.0.1:{PORT}/ -> {BACKEND}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
