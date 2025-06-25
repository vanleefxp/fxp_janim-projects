from pathlib import Path

from janim.imports import *

DIR = Path(__file__).parent if "__file__" in locals() else Path.cwd()

subtitleConfig = {
    "scale": (1, 0.75),
}


class TL_OnePlusOne(Timeline):
    CONFIG = Config(
        font=("Fira Sans", "Source Han Sans SC"),
        typst_shared_preamble="""
        #set text(
            font: ("Fira Sans", "Source Han Sans"),
            lang: "zh",
        )
        #show raw: set text(font: ("Fira Code", "Maple Mono NF"))
        #show math.equation: set text(font: "Fira Math", weight: 400)
        #set par(leading: 1em, spacing: 1.5em, justify: true)
        """,
    )

    def construct(self):
        # 整活视频：如何计算 1 + 1

        # 观前提醒

        i_title = Title("观前提醒", font_size=48)
        i_reminder = (
            TypstDoc((DIR / "assets/reminder_zh.typ").read_text(encoding="utf-8"))
            .points.to_center()
            .shift(DOWN * 0.6)
            .r
        )
        i_title_en = Title("Friendly Reminder", font_size=48, underline_width=13.5)
        i_reminder_en = (
            TypstDoc((DIR / "assets/reminder_en.typ").read_text(encoding="utf-8"))
            .points.to_center()
            .shift(DOWN * 0.6)
            .r
        )
        self.play(Write(i_title), FadeIn(i_reminder))
        self.forward(1.5)
        self.play(FadeOut(Group(i_title, i_reminder)), duration=0.5)
        self.play(Write(i_title_en, duration=1), FadeIn(i_reminder_en))
        self.forward(1.5)
        self.play(FadeOut(Group(i_title_en, i_reminder_en)), duration=0.5)
        self.forward(1)

        # 引言

        i_1plus1 = TypstMath('1 + 1 = "?"').points.scale(4).shift((0, 0.25, 0)).r
        i_think = (
            SVGItem(DIR / "assets/image/thinking.svg")(VItem)
            .color.set(WHITE)
            .r.points.scale(0.5)
            .to_border(DL)
            .r
        )
        i_presentation = (
            SVGItem(DIR / "assets/image/presentation.svg")(VItem)
            .color.set(WHITE)
            .r.points.scale(0.4)
            .to_border(LEFT, buff=1.5)
            .shift(UP * 0.25)
            .r
        )
        i_1plus1Process = (
            TypstMath("""
                &&1 + 1 &= S(0) + S(0) \\
                && &= S(S(0) + 0) \\
                && &= S(S(0))
            """)
            .points.scale(2)
            .shift((2.5, 0.25, 0))
            .r
        )

        self.subtitle(
            (
                "你是否在为忘记如何计算 1 + 1 而苦恼？",
                "Have you ever faced with the dilemma that you have forgotten how to calculate one plus one? ",
            ),
            delay=-0.5,
            duration=4.5,
            **subtitleConfig,
        )
        self.play_audio(Audio(DIR / "assets/audio/1.wav"), delay=-0.5)
        self.prepare(Write(i_1plus1), FadeIn(i_think, scale=0.75))
        self.forward(4.4)

        self.subtitle(
            ("你并非孤身一人。", "You’re not alone."),
            **subtitleConfig,
            duration=1.75,
        )
        self.forward(2)

        self.subtitle(
            (
                "我相信每一位读者都或多或少被这道极其复杂的问题所困扰。",
                "I believe that every reader has been puzzled by this extremely confusing problem. ",
            ),
            **subtitleConfig,
            duration=4.55,
        )
        self.forward(4.8)

        self.subtitle(
            (
                "然而，通过严格的数学推理，",
                "However, under the strict deduction of mathematics, ",
            ),
            **subtitleConfig,
            duration=2.75,
        )
        self.prepare(FadeOut(Group(i_1plus1, i_think)), duration=0.5)
        self.prepare(FadeIn(i_presentation, scale=0.75), Write(i_1plus1Process), at=0.5)
        self.forward(3)

        self.subtitle(
            (
                "我们可以轻松地解开这道难题。",
                "we may have the chance to tackle this difficult problem at ease.",
            ),
            **subtitleConfig,
            duration=3.75,
        )
        self.forward(5)
        self.play(FadeOut(Group(i_presentation, i_1plus1Process)), duration=0.5)

        # 理清概念
        self.play_audio(Audio(DIR / "assets/audio/2.wav"), delay=-0.5)
        self.subtitle(
            (
                "在解决数学问题之前，我们要先理清概念。",
                "Getting clear of the definitions is crucial to solving a mathematic problem. ",
            ),
            **subtitleConfig,
            delay=-0.5,
            duration=4.5,
        )
        self.prepare(FadeIn(i_1plus1, scale=0.75))
        self.forward(4.25)

        self.subtitle(
            (
                "对于计算 1 + 1 而言，其中涉及到的关键概念包括：",
                "For the calculation of one plus one, the key concepts involved are: ",
            ),
            **subtitleConfig,
            duration=4,
        )
        self.forward(4.25)

        i_one = i_1plus1[0]
        i_plus = i_1plus1[1]
        i_questionMarksOne = (
            Text("??", font_size=36)
            .points.next_to(i_one, UP, buff=0.5)
            .r.color.set(YELLOW)
            .r
        )
        i_questionMarksPlus = (
            Text("??", font_size=36)
            .points.next_to(i_plus, DOWN, buff=0.5)
            .r.color.set(YELLOW)
            .r
        )

        self.subtitle(
            (
                "什么是 1, 以及什么是加法",
                "what is “one”, and what is “plus”. ",
            ),
            **subtitleConfig,
            duration=2.5,
        )
        self.prepare(
            i_one.anim.glow.set(color=YELLOW, alpha=0.5, size=0.75),
            FadeIn(i_questionMarksOne),
        )
        self.prepare(
            i_plus.anim.glow.set(color=YELLOW, alpha=0.5, size=0.75),
            FadeIn(i_questionMarksPlus),
            at=1.25,
        )
        self.forward(2.75)

        i_title = Title("Peano Axioms", font_size=48)
        i_peano = (
            TypstDoc((DIR / "assets/peano-axioms.typ").read_text(encoding="utf-8"))
            .points.shift((1, -1.25, 0))
            .r
        )

        self.subtitle(
            (
                "为了回答这两个问题，我们需要引入皮亚诺公理系统",
                "To answer these two questions, we have to introduce Peano axioms, ",
            ),
            **subtitleConfig,
            duration=4.25,
        )
        self.prepare(
            FadeOut(Group(i_questionMarksOne, i_questionMarksPlus, i_1plus1)),
            duration=0.5,
            at=2,
        )
        self.prepare(Write(i_title), at=2.5, duration=1)
        self.prepare(Write(i_peano), at=3.5, duration=3)
        self.forward(4.5)

        self.subtitle(
            (
                "及其对算术运算的定义。",
                "which construct the fundamental framework of natural numbers, ",
            ),
            **subtitleConfig,
            duration=3,
        )
        self.forward(3.25)

        self.subtitle(
            (
                "这一公理系统构建了自然数概念的基本框架。",
                "and its definition of arithmetic computations.",
            ),
            **subtitleConfig,
            duration=3,
        )
        self.forward(5)

        # 介绍皮亚诺公理系统

        i_peano1 = i_peano[:5]
        i_peano6 = i_peano[85:99]
        i_peano8 = i_peano[124:138]
        i_sLetters = i_peano[92, 110, 115, 131, 158]
        i_sNote = (
            TypstText("$S(m)$ means successor of $m$")
            .points.next_to(i_peano6, buff=4)
            .r.color.set(YELLOW)
            .r
        )

        self.play_audio(Audio(DIR / "assets/audio/3.wav"), delay=-0.5)
        self.subtitle(
            (
                "皮亚诺公理首先声明存在一个自然数叫作 0, ",
                "Peano axioms state that we have a natural number called “zero”, ",
            ),
            **subtitleConfig,
            duration=3.5,
            delay=-0.5,
        )
        self.prepare(ShowPassingFlashAround(i_peano1, time_width=5), duration=3)
        self.forward(3.25)

        self.subtitle(
            (
                "又声明了一个函数 $S(m)$ 表示一个自然数的后继",
                "and there is a function $S(m)$ denoting the _successor_ of each natural number. ",
            ),
            **subtitleConfig,
            use_typst_text=True,
            duration=3.75,
        )
        self.prepare(
            *(i_.anim.glow.set(color=YELLOW, alpha=0.5, size=0.5) for i_ in i_sLetters)
        )
        self.prepare(FadeIn(i_sNote), at=0.5)
        self.forward(4)

        self.subtitle(
            (
                "每个自然数都有后继且 0 不是任何自然数的后继",
                "Every natural number has a successor while 0 is not the successor of any other. ",
            ),
            **subtitleConfig,
            use_typst_text=True,
            duration=5.25,
        )
        self.prepare(ShowPassingFlashAround(i_peano6, time_width=5), duration=3)
        self.prepare(
            ShowPassingFlashAround(i_peano8, time_width=5), duration=3, at=2.75
        )
        self.forward(5.5)

        i_successors = (
            TypstMath("&S(0) = 1 quad &&S(1) = 2 \\ &S(2) = 3 quad &&S(3) = 4 quad ...")
            .points.shift((4, 1, 0))
            .r
        )

        self.subtitle(
            (
                "通常，我们将 0 的后继称为 1, ",
                "Typically, the successor of “zero” is referred to as “one”, ",
            ),
            **subtitleConfig,
            use_typst_text=True,
            duration=4,
        )
        self.prepare(Write(i_successors), duration=2)
        self.forward(4.25)

        self.subtitle(
            (
                "1 的后继称为 2.",
                "and the successor of “one” is called “two”.",
            ),
            **subtitleConfig,
            use_typst_text=True,
            duration=3.25,
        )
        self.forward(5)
        self.play(FadeOut(Group(i_successors, i_title, i_peano, i_sNote)), duration=0.5)

        # 定义加法

        i_title = Title("Addition", font_size=48)
        i_addition = (
            TypstMath(
                '&m &&+ &&0 && " " = " " &&m \\ &m &&+ &S(&n) && " " = " " &S(&m + n)'
            )
            .points.scale(2)
            .shift(RIGHT * 0.25)
            .r
        )
        i_addition1 = i_addition[:5]
        i_addition2 = i_addition[5:]

        self.play_audio(Audio(DIR / "assets/audio/4.wav"), delay=-0.5)
        self.subtitle(
            (
                "皮亚诺算术系统基于以下两条规则递归地定义了加法：",
                "Addition in Peano arithmetic system is defined recursively by the following two rules:",
            ),
            **subtitleConfig,
            duration=4.75,
            delay=-0.5,
        )
        self.prepare(Write(i_title))
        self.forward(4.5)

        self.subtitle(
            (
                "任何自然数加 0 等于其自身，",
                "adding zero to any natural number results in itself, ",
            ),
            **subtitleConfig,
            duration=3.25,
        )
        self.prepare(Write(i_addition1))
        self.forward(3.5)

        self.subtitle(
            (
                "以及自然数 $m$ 加上 $n$ 的后继 $S(n)$",
                "and adding a natural number $n$'s successor to another natural number $m$",
            ),
            **subtitleConfig,
            duration=3.49,
            use_typst_text=True,
        )
        self.prepare(Write(i_addition2))
        self.forward(3.5)

        self.subtitle(
            (
                "等于 $(m + n)$ 的后继 $S(m + n)$.",
                "results in the succession of $(m + n)$. ",
            ),
            **subtitleConfig,
            duration=2.5,
            use_typst_text=True,
        )
        self.forward(2.75)

        # 计算 1 + 1

        self.play(FadeOut(Group(i_title, i_addition)), duration=0.5)

        i_title = Title("Calculating 1 + 1", font_size=48)
        i_oneIsS0 = TypstMath("1 = S(0)").points.scale(4).shift(DOWN * 0.25).r
        i_calc = (
            SVGItem(DIR / "assets/image/calculate-1.svg")(VItem)
            .points.scale(0.5)
            .shift(LEFT * 3)
            .r.color.set(WHITE)
            .r.radius.set(0.1)
            .r
        )
        i_1plus1Process = (
            TypstMath("""
                &&1 + 1 &= S(0) + S(0) \\
                && &= S(S(0) + 0) \\
                && &= S(S(0)) \\
                && & = 2
            """)
            .points.shift(RIGHT * 2.5)
            .scale(1.25)
            .r
        )

        self.subtitle(
            (
                "记住了上面这些规则，我们就可以开始着手计算 1 + 1 了",
                "With these rules in mind, we can not get down to the calculation of one plus one.",
            ),
            **subtitleConfig,
            duration=5,
        )
        self.prepare(Write(i_title))
        self.forward(7)

        self.play_audio(Audio(DIR / "assets/audio/5.wav"), delay=-0.5)
        self.subtitle(
            (
                "首先，我们知道 1 被定义为 0 的后继。",
                "First, recall that one is defined to be the successor of zero.",
            ),
            **subtitleConfig,
            duration=4.75,
            delay=-0.5,
        )
        self.prepare(Write(i_oneIsS0))
        self.forward(4.5)

        self.subtitle(
            (
                "因此，我们可以把 1 + 1 写作 $S(0) + S(0)$",
                "Therefore, we can denote $1 + 1$ as $S(0) + S(0)$. ",
            ),
            **subtitleConfig,
            duration=5.25,
            use_typst_text=True,
        )
        self.prepare(FadeOut(i_oneIsS0), duration=0.5)
        self.prepare(FadeIn(i_calc, scale=0.75), at=0.5)
        self.prepare(Write(i_1plus1Process[:4]), at=1.5)
        self.prepare(
            Transform(
                i_1plus1Process[0],
                i_1plus1Process[4:8],
                hide_src=False,
                flatten=True,
                path_arc=-PI / 2,
            ),
            Transform(
                i_1plus1Process[1],
                i_1plus1Process[8],
                hide_src=False,
                path_arc=-PI / 2,
            ),
            Transform(
                i_1plus1Process[2],
                i_1plus1Process[9:13],
                hide_src=False,
                flatten=True,
                path_arc=-PI / 2,
            ),
            at=3.5,
        )
        self.forward(5.5)

        self.subtitle(
            (
                "接下来，根据加法的递归定义，$S(0) + S(0)$ 应等于 $S(S(0) + 0)$",
                "Then, by the recursive definition of addition, $S(0) + S(0)$ equals to $S(S(0) + 0)$.",
            ),
            **subtitleConfig,
            duration=7.75,
            use_typst_text=True,
        )
        self.prepare(
            Transform(i_1plus1Process[4:8], i_1plus1Process[16:20], hide_src=False),
            Transform(i_1plus1Process[3], i_1plus1Process[13], hide_src=False),
            Transform(i_1plus1Process[8], i_1plus1Process[20], hide_src=False),
            Transform(i_1plus1Process[11], i_1plus1Process[21], hide_src=False),
            FadeIn(Group(*i_1plus1Process[14:16], i_1plus1Process[22])),
            at=5,
        )
        self.forward(8)

        self.subtitle(
            (
                "然后，根据递归定义的第一条规则，",
                "Afterwards, by the first statement of the recursive definition, ",
            ),
            **subtitleConfig,
            duration=3.75,
        )
        self.forward(4)

        self.subtitle(
            (
                "任何自然数加上 0 等于其自身，",
                "adding zero to any natural number results in itself, ",
            ),
            **subtitleConfig,
            duration=2.75,
        )
        self.forward(3)

        self.subtitle(
            (
                "因此 $S(0) + 0$ 就等于 $S(0)$",
                "so $S(0) + 0$ simply equals to $S(0)$. ",
            ),
            **subtitleConfig,
            duration=3.75,
            use_typst_text=True,
        )
        self.prepare(
            ShowPassingFlashAround(i_1plus1Process[16:22], time_width=5), duration=4
        )
        self.prepare(
            Transform(i_1plus1Process[16:22], i_1plus1Process[26:30], hide_src=False),
            at=2.5,
        )
        self.forward(4)

        self.subtitle(
            (
                "最终我们得到 $1 + 1 = S(S(0))$",
                "Finally we come to realize that one plus one equals to $S(S(0))$, ",
            ),
            **subtitleConfig,
            duration=4.45,
            use_typst_text=True,
        )
        self.prepare(
            Transform(i_1plus1Process[13:16], i_1plus1Process[23:26], hide_src=False),
            Transform(i_1plus1Process[22], i_1plus1Process[30], hide_src=False),
            at=1,
        )
        self.forward(4.6)

        self.subtitle(
            (
                "也就是 2.",
                "which is — by definition — two.",
            ),
            **subtitleConfig,
            duration=3,
        )
        self.prepare(
            Transform(i_1plus1Process[23], i_1plus1Process[31], hide_src=False),
            Transform(
                i_1plus1Process[24:31],
                i_1plus1Process[32],
                hide_src=False,
                flatten=True,
            ),
            at=0.5,
        )
        self.prepare(
            i_1plus1Process[-1].anim.glow.set(color=YELLOW, alpha=0.5, size=0.5), at=1.5
        )
        self.forward(4)

        # 总结

        self.play_audio(Audio(DIR / "assets/audio/6.wav"))
        self.subtitle(
            (
                "哇！至此我们已经成功解决了这道世纪难题！",
                "Wow! Up till now we have successfully conquered the century-old computational challenge. ",
            ),
            **subtitleConfig,
            duration=6.25,
        )
        self.forward(6.5)

        self.subtitle(
            (
                "真的是太棒啦！",
                "So great that we did it!",
            ),
            **subtitleConfig,
            duration=2,
        )
        self.forward(2.75)

        self.play(FadeOut(Group(i_1plus1Process, i_calc, i_title)))
        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
