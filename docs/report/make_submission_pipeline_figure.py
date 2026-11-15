"""Vector-only inference diagram; all miniature masks are labeled schematics.
No observations, predictions, annotations, fitting, or model execution are read.
"""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, PathPatch
from matplotlib.path import Path as MPath
import numpy as np

OUT = Path(__file__).resolve().parent
INK = '#172D3B'
MUTED = '#526775'
LINE = '#98A9B3'
BLUE = '#277BA8'
TEAL = '#16857F'
GOLD = '#B97A26'
VIOLET = '#7564A9'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                     'mathtext.fontset': 'dejavusans'})
fig = plt.figure(figsize=(7.0, 2.25), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set(xlim=(0, 720), ylim=(0, 230)); ax.axis('off')

def text(x,y,s,size=7.5,color=INK,weight='normal',ha='left',**kwargs):
    return ax.text(x,y,s,fontsize=size,color=color,fontweight=weight,ha=ha,va='center',**kwargs)

def card(x,y,w,h,color='#F6F8FA',edge='#DDE5EA',radius=6):
    return ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                       facecolor=color,edgecolor=edge,linewidth=.65))

def arrow(points,color=LINE,lw=.9,dashed=False):
    points=np.asarray(points)
    ax.add_patch(FancyArrowPatch(path=MPath(points,[MPath.MOVETO]+[MPath.LINETO]*(len(points)-1)),
      arrowstyle='-|>',mutation_scale=7.5,color=color,linewidth=lw,
      linestyle=(0,(2.4,2.4)) if dashed else '-',joinstyle='round'))

def label(n,x,title,color=INK):
    ax.add_patch(Circle((x+6,213),7.5,facecolor=color,edgecolor='none'))
    text(x+6,213,n,6.8,'white','bold','center')
    text(x+19,213,title,8.2,INK,'bold')

def curve(points,color,lw=1.3,alpha=1,linestyle='-'):
    ax.add_patch(PathPatch(MPath(points,[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4]),
       fill=False,edgecolor=color,linewidth=lw,alpha=alpha,capstyle='round',linestyle=linestyle))

# Shared input: an explicitly schematic H-alpha disk, drawn as vector primitives.
text(40,184,'Native image',6.8,INK,'bold','center')
for radius in np.linspace(27,1,40):
    shade=.57+.25*(1-(radius/27)**2)
    ax.add_patch(Circle((40,133),radius,facecolor=(shade,shade,shade),edgecolor='none'))
curve([(24,134),(34,141),(30,124),(47,128)],'#48545B',1.7)
curve([(39,149),(48,144),(43,140),(54,136)],'#4F5B61',1.3)
curve([(30,117),(36,120),(40,115),(48,119)],'#5E686D',1.1)
text(40,93,r'H$\alpha$',8,INK,ha='center')
text(40,78,r'$2048\times2048$',7.2,MUTED,ha='center')

label('1',93,'Parallel models',BLUE)
card(94,124,150,70,'#EBF6F3','#D4E8E2')
card(94,43,150,70,'#EDF5FA','#D6E5EE')
text(104,179,'B5 + ConvNeXt-Tiny',7.4,TEAL,'bold')
text(104,160,'U-Nets · tiled · 4-view TTA',6.5)
text(104,142,'Equal probability mean',6.8,MUTED)
text(104,97,'YOLO26s-seg',8.3,BLUE,'bold')
text(104,78,r'1536 input · $c\geq0.03$',7.1)
text(104,61,'Native instance masks',7.0,MUTED)
text(104,49,'Box + mask NMS: 0.5',6.6,MUTED)
arrow([(69,133),(80,133),(80,159),(92,159)],TEAL)
arrow([(80,133),(80,78),(92,78)],BLUE)

label('2',272,'Refine boundaries',TEAL)
card(272,43,146,151,'#F7F9FA')
text(345,178,'Combine both signals',7.1,INK,'bold','center')
# Support envelope, raw proposal, and refined boundary: schematic geometry.
curve([(299,135),(318,159),(340,106),(386,136)],'#D6E3E7',11)
curve([(299,135),(318,159),(340,106),(386,136)],BLUE,5.4,.27)
curve([(300,137),(321,152),(337,112),(384,134)],TEAL,3.2)
text(345,106,'Local support: R = 12 px',6.9,MUTED,ha='center')
text(345,84,r'$0.75P+0.25G_{\sigma=3}(m)$',8.0,INK,ha='center')
text(345,65,r'Threshold $\geq0.5$',7.0,INK,ha='center')
text(345,50,'Then mask NMS: IoU 0.5',6.6,MUTED,ha='center')
arrow([(245,159),(257,159),(257,143),(270,143)],TEAL)
arrow([(245,78),(257,78),(257,94),(270,94)],BLUE)
text(261,168,r'$P$',6.5,TEAL,ha='center')
text(262,70,r'$m$',6.5,BLUE,ha='center')

label('3',446,'Quality gate',GOLD)
card(446,43,111,151,'#FBF8F2','#EBE3D7')
text(501.5,178,'Frozen HGB',7.4,INK,'bold','center')
text(501.5,160,'27 input features',6.8,MUTED,ha='center')
# Illustrative utility gate (no measured scores implied).
for y,width,color in [(130,46,GOLD),(114,31,GOLD),(98,11,'#C6CACB')]:
    card(464,y-4,width,8,color,color,2)
ax.plot([486,486],[88,140],color=GOLD,linewidth=.8,linestyle=(0,(2,2)))
text(501.5,72,r'Keep $q\geq0.20$',8.0,INK,'bold','center')
text(501.5,53,'No test-time fitting',6.6,MUTED,ha='center')
arrow([(419,119),(444,119)],MUTED)

label('4',585,'Disjoint export',VIOLET)
card(585,43,133,151,'#F5F3F9','#E1DDEA')
text(651.5,178,'Order by confidence',7.0,INK,'bold','center')
text(651.5,160,'Subtract owned pixels',7.0,MUTED,ha='center')
# Truly disjoint vector contours from a synthetic pair, not a prediction.
x,y=np.meshgrid(np.linspace(-1,1,180),np.linspace(-1,1,100))
a=(np.abs(y-(.18*np.sin(4*x)+.15*x))<.12)&(np.abs(x)<.92)
b=(np.abs(y-(-.55*x+.06*np.cos(4*x)))<.12)&(np.abs(x)<.7)
b=b&~a
assert not np.any(a&b)
xx=651.5+x*43; yy=123+y*24
ax.contourf(xx,yy,a.astype(float),levels=[.5,1.5],colors=[BLUE])
ax.contourf(xx,yy,b.astype(float),levels=[.5,1.5],colors=[VIOLET])
text(651.5,84,r'Remainder area $\geq5$ px',6.9,INK,ha='center')
text(651.5,64,'Native COCO RLEs',7.5,INK,'bold','center')
text(651.5,50,'Zero shared pixels',6.6,MUTED,ha='center')
arrow([(558,119),(583,119)],MUTED)

# Confidence is preserved through refinement and quality selection, not replaced by q.
arrow([(208,42),(208,20),(651.5,20),(651.5,41)],MUTED,.75,True)
ax.add_patch(FancyBboxPatch((287,10),282,18,boxstyle='square,pad=0',facecolor='white',edgecolor='none'))
text(428,19,'Original detector identity and confidence retained',6.7,MUTED,ha='center')
text(716,5,'Mask illustrations are schematic',5.7,MUTED,ha='right')

fig.canvas.draw()
renderer=fig.canvas.get_renderer()
frame=fig.bbox
for item in ax.texts:
    bounds=item.get_window_extent(renderer)
    assert bounds.x0>=frame.x0 and bounds.y0>=frame.y0 and bounds.x1<=frame.x1 and bounds.y1<=frame.y1, ('Clipped text',item.get_text(),bounds)

for suffix in ('pdf','svg','png'):
    fig.savefig(OUT/f'own_hybrid_submission_pipeline.{suffix}',dpi=240,transparent=False)
plt.close(fig)
files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),*[OUT/f'own_hybrid_submission_pipeline.{s}' for s in ('pdf','svg','png')]]}
(OUT/'own_hybrid_submission_pipeline_provenance.json').write_text(json.dumps({
    'description':'Vector inference diagram with synthetic, explicitly labeled illustrative masks.',
    'data_or_predictions_accessed':False,'model_execution':False,'fits':0,
    'frozen_quality_model_at_inference':True,'synthetic_export_overlap_pixels':0,
    'files_sha256':files},indent=2)+'\n')
print(json.dumps({'outputs':list(files),'synthetic_overlap':0}))
