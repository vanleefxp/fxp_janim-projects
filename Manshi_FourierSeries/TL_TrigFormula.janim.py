from pathlib import Path

from janim.imports import *

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()
config = Config(
    font=[
        "NewComputerModern10",
        "FandolSong",
    ],
    typst_shared_preamble=(DIR / "../assets/typst/manshi_preamble.typ").read_text(),
)


class TL_TrigFormula(Timeline):
    CONFIG = config

    def construct(self):
        i_trigProd2Sum = (
            TypstDoc(
                (DIR / "assets/typst/trig_prod2sum.typ").read_text(encoding="utf-8")
            )
            .points.to_center()
            .shift(UP * 0.25)
            .r
        )
        i_trigIntegral = (
            TypstDoc(
                (DIR / "assets/typst/trig_integral.typ").read_text(encoding="utf-8")
            )
            .points.to_center()
            .shift(UP * 0.25)
            .r
        )
        self.play(Write(i_trigProd2Sum))
        self.forward(2)
        self.play(FadeOut(i_trigProd2Sum))
        self.play(Write(i_trigIntegral))
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
