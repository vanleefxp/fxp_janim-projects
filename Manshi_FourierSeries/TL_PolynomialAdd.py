from janim.imports import *
from frozendict import frozendict
from common import *

with reloads():
    from common import *


class TL_PolynomialAdd(Timeline):
    def construct(self):
        class PolySumTermOpt:
            def __init__(self, order):
                self.order = order
                self.sign = 1
                self.coefSrc = f"(a_{order} + b_{order})"
                self.omit = False

        class PolyMulTermOpt:
            def __init__(self, order):
                self.order = order
                self.sign = 1
                self.coefSrc = f"k a_{order}"
                self.omit = False

        polyWidths = frozendict(coef=1.6)
        polyAligns = frozendict(coef=0)
        textStrokeCfg = frozendict(stroke_alpha=1, stroke_radius=0.005)
        i_poly1 = PolynomialText(
            terms=(NumberedPolyTermOpt(i, "a") for i in range(4)),
            nameSymbol="p(x)",
            widths=polyWidths,
            aligns=polyAligns,
        )
        i_poly2 = (
            PolynomialText(
                terms=(NumberedPolyTermOpt(i, "b") for i in range(4)),
                nameSymbol="q(x)",
                widths=polyWidths,
                aligns=polyAligns,
            )
            .points.shift(DOWN * 0.8)
            .r
        )
        i_polySum = (
            PolynomialText(
                terms=(PolySumTermOpt(i) for i in range(4)),
                nameSymbol="p(x) + q(x)",
                widths=polyWidths,
                aligns=polyAligns,
            )
            .points.shift(DOWN * 1.6)
            .r
        )
        i_polyMul = (
            PolynomialText(
                terms=(PolyMulTermOpt(i) for i in range(4)),
                nameSymbol="k p(x)",
                widths=polyWidths,
                aligns=polyAligns,
            )
            .points.shift(DOWN * 2.4)
            .r
        )

        vec_p = np.array((1.2, 1.1))
        vec_q = np.array((1.6, -0.6))
        vec_sum = vec_p + vec_q
        vecScale = 1.8

        i_vpTem = Vector(vec_p, color=RED, tip_kwargs=arrowCfg)
        i_vqTem = Vector(vec_q, color=GREEN, tip_kwargs=arrowCfg)

        i_vp1 = i_vpTem.copy().points.shift((-6, -1.25, 0)).r
        i_vq1 = i_vqTem.copy().points.shift((-6, -1.5, 0)).r

        vec2Pos = np.array((-2, -1.375, 0))
        i_vp2 = i_vpTem.copy().points.shift(vec2Pos).r
        i_vq2 = i_vqTem.copy().points.shift(vec2Pos).r
        i_vSum = (
            Vector(vec_sum, color=BLUE, tip_kwargs=arrowCfg).points.shift(vec2Pos).r
        )
        i_lineToVp = Line(
            i_vp2.points.get_end(),
            i_vSum.points.get_end(),
            stroke_alpha=0.5,
            stroke_radius=0.015,
            depth=1,
        )
        i_lineToVq = Line(
            i_vq2.points.get_end(),
            i_vSum.points.get_end(),
            stroke_alpha=0.5,
            stroke_radius=0.015,
            depth=1,
        )

        vec3Pos = np.array((3, -2.25, 0))
        i_vp3 = i_vpTem.copy().points.shift(vec3Pos).r
        i_vMul = (
            Vector(vec_p * vecScale, color=PINK, depth=1, tip_kwargs=arrowCfg)
            .points.shift(vec3Pos)
            .r
        )

        i_vpText1 = (
            TypstMath("p(x)", **textStrokeCfg)
            .points.next_to(i_vp1.points.get_end(), RIGHT, buff=0.1)
            .r
        )
        i_vqText1 = (
            TypstMath("q(x)", **textStrokeCfg)
            .points.next_to(i_vq1.points.get_end(), RIGHT, buff=0.1)
            .r
        )
        i_vpText2 = (
            i_vpText1.copy().points.next_to(i_vp2.points.get_end(), UL, buff=0.05).r
        )
        i_vqText2 = (
            i_vqText1.copy().points.next_to(i_vq2.points.get_end(), DOWN, buff=0.1).r
        )
        i_vSumText = (
            TypstMath("p(x) + q(x)", **textStrokeCfg)
            .points.next_to(i_vSum.points.get_end(), RIGHT, buff=0.1)
            .r
        )
        i_vpText3 = (
            i_vpText1.copy().points.next_to(i_vp3.points.get_end(), DR, buff=0.05).r
        )
        i_vMulText = (
            TypstMath("k p(x)", **textStrokeCfg)
            .points.next_to(i_vMul.points.get_end(), DR, buff=0.05)
            .r
        )

        # 着色
        i_poly1.i_symbol[0].set(color=RED)  # p
        i_poly2.i_symbol[0].set(color=GREEN)  # q
        i_polySum.i_symbol[0].set(color=RED)  # p
        i_polySum.i_symbol[5].set(color=GREEN)  # q
        i_polyMul.i_symbol[0].set(color=PINK)  # k
        i_polyMul.i_symbol[1].set(color=RED)  # p

        for i_coef in i_poly1.i_coefs:
            i_coef[0].set(color=RED)  # a

        for i_coef in i_poly2.i_coefs:
            i_coef[0].set(color=GREEN)  # b

        for i_coef1, i_coefSum in zip(i_poly1.i_coefs, i_polySum.i_coefs):
            i_coefSum[1].set(color=RED)  # a
            i_coefSum[len(i_coef1) + 2].set(color=GREEN)

        for i_coef in i_polyMul.i_coefs:
            i_coef[0].set(color=PINK)  # k
            i_coef[1].set(color=RED)  # a

        Group(i_poly1, i_poly2, i_polySum, i_polyMul).set(
            stroke_alpha=1, stroke_radius=0.005
        ).points.scale(0.95).to_center().shift((-0.25, 2, 0))
        self.play(Write(i_poly1), Write(i_poly2), duration=1, lag_ratio=0.5)
        self.forward(0.5)
        self.play(
            AnimGroup(
                getVecCreateAnim(i_vp1, duration=1),
                FadeIn(i_vpText1, duration=0.5),
                lag_ratio=0.5,
            ),
            AnimGroup(
                getVecCreateAnim(i_vq1, duration=1),
                FadeIn(i_vqText1, duration=0.5),
                lag_ratio=0.5,
            ),
            lag_ratio=0.5,
        )
        self.forward(1)

        # 两个多项式相加
        self.play(
            Transform(
                i_poly1.i_symbol,
                i_polySum.i_symbol[: len(i_poly1.i_symbol)],
                hide_src=False,
            ),  # p(x)
            Transform(
                i_poly2.i_symbol,
                i_polySum.i_symbol[-len(i_poly2.i_symbol) :],
                hide_src=False,
            ),  # q(x)
            FadeIn(i_polySum.i_eq),  # =
            FadeIn(
                i_polySum.i_symbol[len(i_poly1.i_symbol) : -len(i_poly2.i_symbol)]
            ),  # +
        )
        self.forward(0.5)
        self.play(
            Transform(i_vp1, i_vp2, hide_src=False),
            Transform(i_vq1, i_vq2, hide_src=False),
            Transform(i_vpText1, i_vpText2, hide_src=False),
            Transform(i_vqText1, i_vqText2, hide_src=False),
        )
        self.forward(0.5)

        for (
            i_coef1,
            i_coef2,
            i_coefSum,
            i_termSum,
            i_signSum,
        ) in zip(
            i_poly1.i_coefs,
            i_poly2.i_coefs,
            i_polySum.i_coefs,
            i_polySum.i_terms,
            i_polySum.i_signs,
        ):
            self.play(
                Transform(
                    i_coef1, i_coefSum[1 : len(i_coef1) + 1], hide_src=False
                ),  # a_i
                Transform(
                    i_coef2, i_coefSum[-len(i_coef2) - 1 : -1], hide_src=False
                ),  # b_i
                *(
                    FadeIn(item)
                    for item in (
                        i_coefSum[0],
                        i_coefSum[-1],
                        i_coefSum[len(i_coef1) + 1],
                        i_termSum,
                        i_signSum,
                    )
                ),
                duration=0.75,
            )
            self.forward(0.25)
        self.play(
            *(FadeIn(item) for item in (i_polySum.i_signs[-1], i_polySum.i_ellipsis)),
            duration=0.75,
        )
        self.forward(0.5)
        self.play(
            Transform(i_vp2, i_vSum, hide_src=False),
            Transform(i_vq2, i_vSum, hide_src=False),
            Create(i_lineToVp),
            Create(i_lineToVq),
        )
        self.play(Write(i_vSumText))
        self.forward(1)

        # 多项式数乘
        self.play(
            Transform(i_poly1.i_symbol, i_polyMul.i_symbol[1:], hide_src=False),  # p(x)
            Transform(i_poly1.i_eq, i_polyMul.i_eq, hide_src=False),  # =
            FadeIn(i_polyMul.i_symbol[0]),  # k
        )
        self.forward(0.5)
        self.play(
            Transform(i_vp1, i_vp3, hide_src=False),
            FadeIn(i_vpText3, duration=0.5),
            lag_ratio=0.75,
        )
        self.forward(0.5)
        for i_coef1, i_coefMul, i_termMul, i_signMul in zip(
            i_poly1.i_coefs, i_polyMul.i_coefs, i_polyMul.i_terms, i_polyMul.i_signs
        ):
            self.play(
                Transform(i_coef1, i_coefMul[1:], hide_src=False),  # a_i
                *(FadeIn(item) for item in (i_termMul, i_signMul, i_coefMul[0])),
                duration=0.75,
            )
            self.forward(0.25)
        self.play(
            *(FadeIn(item) for item in (i_polyMul.i_signs[-1], i_polyMul.i_ellipsis)),
            duration=0.75,
        )
        self.forward(0.5)
        self.play(
            Transform(i_vp3, i_vMul, hide_src=False),
            Transform(i_vpText3, i_vMulText[1:], hide_src=False),
            FadeIn(i_vMulText[0]),
        )

        def polyCoefsToMatrix(i_poly: PolynomialText, coefProc=lambda item: item):
            i_coefs = Group(
                *map(
                    lambda item: coefProc(item),
                    i_poly.i_coefs.copy(),
                ),
                i_poly.i_ellipsis.copy().points.rotate(PI / 2).r,
            )
            return TypstMatrix(map(lambda item: (item,), i_coefs)), i_coefs

        i_arrowTem = TypstMath("->", **textStrokeCfg)

        i_poly1SymCp = i_poly1.i_symbol.copy()
        i_arrow1 = i_arrowTem.copy().points.next_to(i_poly1SymCp, RIGHT, buff=0.1).r
        i_poly1VecText, i_poly1VecCoefs = polyCoefsToMatrix(i_poly1)
        i_poly1VecText.points.next_to(i_arrow1, RIGHT, buff=0.1).r
        i_poly1VecGroup = Group(i_poly1SymCp, i_arrow1, i_poly1VecText)

        i_poly2SymCp = i_poly2.i_symbol.copy()
        i_arrow2 = i_arrowTem.copy().points.next_to(i_poly2SymCp, RIGHT, buff=0.1).r
        i_poly2VecText, i_poly2VecCoefs = polyCoefsToMatrix(i_poly2)
        i_poly2VecText.points.next_to(i_arrow2, RIGHT, buff=0.1).r
        i_poly2VecGroup = (
            Group(i_poly2SymCp, i_arrow2, i_poly2VecText)
            .points.next_to(i_poly1VecGroup, RIGHT, buff=0.75)
            .r
        )

        i_polySumSymCp = i_polySum.i_symbol.copy()
        i_arrowSum = i_arrowTem.copy().points.next_to(i_polySumSymCp, RIGHT, buff=0.1).r
        i_polySumVecText, i_polySumVecCoefs = polyCoefsToMatrix(
            i_polySum, coefProc=lambda item: item[1:-1]
        )
        i_polySumVecText.points.next_to(i_arrowSum, RIGHT, buff=0.1).r
        i_polySumGroup = (
            Group(i_polySumSymCp, i_arrowSum, i_polySumVecText)
            .points.next_to(i_poly2VecGroup, RIGHT, buff=0.75)
            .r
        )

        i_polyMulSymCp = i_polyMul.i_symbol.copy()
        i_arrowMul = i_arrowTem.copy().points.next_to(i_polyMulSymCp, RIGHT, buff=0.1).r
        i_polyMulVecText, i_polyMulVecCoefs = polyCoefsToMatrix(i_polyMul)
        i_polyMulVecText.points.next_to(i_arrowMul, RIGHT, buff=0.1).r
        i_polyMulVecGroup = (
            Group(i_polyMulSymCp, i_arrowMul, i_polyMulVecText)
            .points.next_to(i_polySumGroup, RIGHT, buff=0.75)
            .r
        )

        (
            Group(i_poly1VecGroup, i_poly2VecGroup, i_polySumGroup, i_polyMulVecGroup)
            .points.to_center()
            .shift((-0.25, 2, 0))
            .r
        )
        # self.show(i_vecs)
        self.forward(1)
        self.play(
            Transform(i_poly1.i_symbol, i_poly1SymCp),
            Transform(i_poly1.i_eq, i_arrow1),
            # Transform(i_poly1.i_ellipsis, i_poly1VecText[-1], flatten=True),
            Transform(i_poly2.i_symbol, i_poly2SymCp),
            Transform(i_poly2.i_eq, i_arrow2),
            Transform(i_polySum.i_symbol, i_polySumSymCp),
            Transform(i_polySum.i_eq, i_arrowSum),
            Transform(i_polyMul.i_symbol, i_polyMulSymCp),
            Transform(i_polyMul.i_eq, i_arrowMul),
            *(
                FadeOut(getattr(i_poly, attr), duration=0.25)
                for i_poly in (i_poly1, i_poly2, i_polySum, i_polyMul)
                for attr in ("i_signs", "i_terms")
            ),
            *(
                FadeIn(item, duration=0.5)
                for item in (
                    i_poly1VecText[:10],
                    i_poly1VecText[-10:],
                    i_poly2VecText[:10],
                    i_poly2VecText[-10:],
                    i_polySumVecText[:10],
                    i_polySumVecText[-10:],
                    i_polyMulVecText[:10],
                    i_polyMulVecText[-10:],
                )
            ),
            Transform(i_poly1.i_coefs, i_poly1VecCoefs[:-1]),
            Transform(i_poly1.i_ellipsis, i_poly1VecCoefs[-1], flatten=True),
            Transform(i_poly2.i_coefs, i_poly2VecCoefs[:-1]),
            Transform(i_poly2.i_ellipsis, i_poly2VecCoefs[-1], flatten=True),
            Transform(i_polyMul.i_coefs, i_polyMulVecCoefs[:-1]),
            Transform(i_polyMul.i_ellipsis, i_polyMulVecCoefs[-1], flatten=True),
            *(
                FadeOut(item, duration=0.25)
                for item in it.chain(
                    (item1[0] for item1 in i_polySum.i_coefs),
                    (item1[-1] for item1 in i_polySum.i_coefs),
                )
            ),
            Transform(
                Group(*(item[1:-1] for item in i_polySum.i_coefs)),
                i_polySumVecCoefs[:-1],
            ),
            Transform(i_polySum.i_ellipsis, i_polySumVecCoefs[-1], flatten=True),
            duration=4,
        )

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
