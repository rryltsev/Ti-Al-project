from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
builder=HERE.parent/'build_figure.py'
source=builder.read_text(encoding='utf-8')
anchor='plt.close(fig)'
assert source.count(anchor)==1
extra="""axs[0].set_visible(False)
fig.canvas.draw()
bbox=axs[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).expanded(1.045,1.06)
for extension in ['png','pdf','svg']:
 fig.savefig(OUTPUT/f'figure7b_current_campaign.{extension}',dpi=300,bbox_inches=bbox)
plt.close(fig)"""
source=source.replace(anchor,extra)
sys.argv=[str(builder),'--profile',str(HERE/'figure_profile.json'),'--output-dir',str(HERE)]
exec(compile(source,str(builder),'exec'),{'__file__':str(builder),'__name__':'__main__'})
