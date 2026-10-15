"""Build a local animated review of every ordered route (no server required)."""
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[1]
out=root/'fresh_output'
records=[]
for file in sorted((root/'images').glob('*.png')):
    result=json.loads((out/f'{file.stem}.json').read_text())
    records.append(dict(name=file.stem,image='data:image/png;base64,'+base64.b64encode(file.read_bytes()).decode(),**result))
for batch in range(4):
    sheet=Image.new('RGB',(1100,880),'white');draw=ImageDraw.Draw(sheet)
    for i, r in enumerate(records[batch*25:(batch+1)*25]):
        x,y=i%5*220,i//5*176
        im=Image.open(out/f"{r['name']}_overlay.png").resize((220,154))
        sheet.paste(im,(x,y+20))
        draw.text((x+5,y+3),f"{r['name']} {r['chain_coverage']:.0%}",fill='black')
    sheet.save(out/f'results_{batch+1}.png')
html='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Обводка: проверка маршрута</title>
<style>body{font:16px system-ui;margin:24px;background:#f6f6f3;color:#222}button,select,input{font:inherit;min-height:44px}canvas{display:block;background:white;max-width:100%;width:880px;image-rendering:auto;border:1px solid #ccc;margin:20px 0}p{max-width:880px}button{padding:8px 16px}#info{white-space:pre-wrap}</style>
<h1>Обводка по линии</h1><p>Синий → красный: порядок движения. Зелёная точка — старт. Ползунок показывает путь до выбранного момента. У стрелки можно увидеть возврат по тому же крылу. Исходная картинка неизменна.</p>
<select id="figure" aria-label="Фигура"></select> <button id="play">Пауза</button> <button id="restart">Сначала</button> <label><input id="vector" type="checkbox"> Вектор вместо пиксельного пути</label>
<canvas id="canvas" width="880" height="616"></canvas><input id="time" type="range" min="0" max="1000" value="0" style="width:880px;max-width:100%" aria-label="Ход обводки"><p id="info"></p>
<script>const data=__DATA__;const select=document.querySelector('#figure'),canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),slider=document.querySelector('#time');let running=true,progress=0,last=0,img=new Image();data.forEach((r,i)=>select.add(new Option(r.name,i)));function load(){progress=0;const r=data[+select.value];img.src=r.image;document.querySelector('#info').textContent=`${r.name}: ${r.status}\nПокрытие цепочек: ${(100*r.chain_coverage).toFixed(1)}%; отрезков точного вектора: ${r.vector_segments}\nВектор на чернилах: ${r.all_vector_segments_on_ink}; отдельные пиксели чернил: ${r.disconnected_ink_pixels}`;}select.onchange=load;document.querySelector('#play').onclick=()=>{running=!running;document.querySelector('#play').textContent=running?'Пауза':'Продолжить';};document.querySelector('#restart').onclick=()=>progress=0;slider.oninput=()=>{progress=+slider.value/1000;running=false;document.querySelector('#play').textContent='Продолжить';};function frame(t){if(running)progress=Math.min(1,progress+(t-last)/6500);last=t;slider.value=progress*1000;ctx.clearRect(0,0,880,616);if(img.complete&&img.naturalWidth){ctx.drawImage(img,0,0,880,616);const r=data[+select.value],p=document.querySelector('#vector').checked?r.vector:r.path;let lengths=[0];for(let i=1;i<p.length;i++)lengths.push(lengths[i-1]+Math.hypot(p[i][0]-p[i-1][0],p[i][1]-p[i-1][1]));const end=progress*lengths.at(-1);let tip=p[0];for(let i=1;i<p.length&&lengths[i-1]<end;i++){let f=Math.min(1,(end-lengths[i-1])/(lengths[i]-lengths[i-1]||1));tip=[p[i-1][0]+f*(p[i][0]-p[i-1][0]),p[i-1][1]+f*(p[i][1]-p[i-1][1])];ctx.strokeStyle=`hsl(${210-210*lengths[i]/lengths.at(-1)} 85% 43%)`;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(...p[i-1].map(v=>v*4));ctx.lineTo(...tip.map(v=>v*4));ctx.stroke();}ctx.fillStyle='#008448';ctx.beginPath();ctx.arc(...p[0].map(v=>v*4),5,0,7);ctx.fill();ctx.fillStyle='#db3030';ctx.beginPath();ctx.arc(...tip.map(v=>v*4),4,0,7);ctx.fill();}requestAnimationFrame(frame);}load();requestAnimationFrame(frame);</script></html>'''
(out/'review.html').write_text(html.replace('__DATA__',json.dumps(records,separators=(',',':'))))
print(out/'review.html')
