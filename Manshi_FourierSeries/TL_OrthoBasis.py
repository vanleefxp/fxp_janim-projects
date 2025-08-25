from frozendict import frozendict

from janim.imports import *

with reloads():
    from common import *


class TL_OrthoBasis(Timeline):
    CONFIG = config

    def construct(self):
        i_coord = (
            NumberPlane(
                x_range=(-6, 10, 1),
                y_range=(-3, 6, 1),
                background_line_style=dict(stroke_alpha=0.75),
                # x_axis_config=dict(unit_size=0.75),
                # y_axis_config=dict(unit_size=0.75),
                depth=2,
            )
            .points.shift((-2, -1.25, 0))
            .r
        )
        vec = np.array((4, 3))
        preamble1 = """
            #show "u": set text(fill: RED)
            #show "v": set text(fill: GREEN)
        """
        preamble = f"""
            #show sym.alpha: set text(fill: PINK)
            {preamble1}
        """
        i_rightAngle = Elbow(stroke_radius=0.015).mark.set(i_coord.get_origin()).r

        i_ux = Arrow(
            i_coord.get_origin(),
            i_coord.c2p(1, 0),
            buff=0,
            color=RED,
            tip_kwargs=arrowCfg,
            depth=0,
        )
        i_uy = Arrow(
            i_coord.get_origin(),
            i_coord.c2p(0, 1),
            buff=0,
            color=GREEN,
            tip_kwargs=arrowCfg,
            depth=0,
        )
        i_vec = Arrow(
            i_coord.get_origin(),
            i_coord.c2p(*vec),
            buff=0,
            color=PINK,
            tip_kwargs=arrowCfg,
        )

        i_projX = Line(
            i_coord.get_origin(),
            i_coord.c2p(vec[0], 0),
            buff=0,
            color=RED,
            stroke_alpha=0.75,
            depth=1,
        )
        i_projY = Line(
            i_coord.get_origin(),
            i_coord.c2p(0, vec[1]),
            buff=0,
            color=GREEN,
            stroke_alpha=0.75,
            depth=1,
        )

        lineToAxisCfg = frozendict(
            stroke_alpha=0.5,
            stroke_radius=0.015,
            depth=1,
        )
        i_lineToX = Line(i_coord.c2p(*vec), i_coord.c2p(vec[0], 0), **lineToAxisCfg)
        i_lineToY = Line(i_coord.c2p(*vec), i_coord.c2p(0, vec[1]), **lineToAxisCfg)

        i_uxText = (
            TypstMath("boldup(u)").points.next_to(i_coord.c2p(1, 0), DOWN, buff=0.2).r
        )
        i_uyText = (
            TypstMath("boldup(v)").points.next_to(i_coord.c2p(0, 1), LEFT, buff=0.2).r
        )
        i_vecText = (
            TypstMath(
                f"boldup(alpha) = (boldup(u), boldup(v)) vec({vec[0]}, {vec[1]})",
                preamble=preamble1,
            )
            .points.next_to(i_coord.c2p(*vec), RIGHT, buff=0.1)
            .r
        )
        i_prodXText = (
            TypstMath(
                f"(angle.l boldup(u), boldup(alpha) angle.r) / abs(boldup(u))^2 = boldup(alpha)_boldup(u) = {vec[0]}",
                preamble=preamble,
            )
            .points.next_to(i_coord.c2p(vec[0], 0), DOWN, buff=0.2)
            .r
        )
        i_prodYText = (
            TypstMath(
                f"(angle.l boldup(v), boldup(alpha) angle.r) / abs(boldup(v))^2 = boldup(alpha)_boldup(v) = {vec[1]}",
                preamble=preamble,
            )
            .points.rotate(PI / 2)
            .next_to(i_coord.c2p(0, vec[1]), LEFT, buff=0.2)
            .r
        )
        i_uVecText = addBgRect(
            TypstMath(
                """
                boldup(u) = vec(1, 0) quad
                boldup(v) = vec(0, 1)
                """,
                preamble=preamble,
            )
            .points.to_border(UL, buff=0.25)
            .r
        )
        i_uDotText = addBgRect(
            TypstMath("angle.l boldup(u), boldup(v) angle.r = 0", preamble=preamble)
            .points.next_to(i_uVecText, DOWN, buff=0.25, aligned_edge=LEFT)
            .r
        )

        for item in (i_prodXText, i_prodYText, i_vecText):
            addBgRect(item)

        mat = np.array(((1, 0), (1, 1))).T
        self.play(Create(i_coord))

        passingFlashCfg = frozendict(
            stroke_color=YELLOW,
            depth=-1,
            stroke_alpha=1,
            stroke_radius=0.02,
            glow_alpha=0.25,
            glow_color=YELLOW,
            glow_size=0.5,
        )
        self.play(
            *(
                ShowPassingFlash(i_.copy().set(**passingFlashCfg), time_width=1.5)
                for i_ in i_coord.get_axes()
            ),
            lag_ratio=0.25,
            duration=2,
        )
        self.play(Flash(i_coord.get_origin()))
        self.play(FadeIn(i_rightAngle, duration=0.5))
        self.forward(0.5)
        self.play(
            AnimGroup(
                getVecCreateAnim(i_ux, duration=0.5), Write(i_uxText, duration=0.5)
            ),
            AnimGroup(
                getVecCreateAnim(i_uy, duration=0.5), Write(i_uyText, duration=0.5)
            ),
            Write(i_uVecText, duration=1.5),
            FadeIn(i_uDotText),
            lag_ratio=0.5,
        )
        self.forward(0.5)
        self.play(getVecCreateAnim(i_vec, duration=1))
        self.play(Write(i_vecText, duration=0.5))
        self.forward(0.5)
        self.play(
            Transform(i_vec, i_projX, root_only=True, hide_src=False),
            Create(i_lineToX),
            duration=1,
        )
        self.play(Write(i_prodXText), duration=0.75)
        self.forward(0.5)
        self.play(
            Transform(i_vec, i_projY, root_only=True, hide_src=False),
            Create(i_lineToY),
            duration=1,
        )
        self.play(Write(i_prodYText), duration=0.75)

        glowCfg = frozendict(size=0.5, alpha=0.5, color=GREEN_SCREEN)
        self.play(
            i_vecText[-3].anim.glow.set(**glowCfg),
            i_prodXText[-1].anim.glow.set(**glowCfg),
            duration=0.25,
        )
        self.forward(0.75)
        self.play(
            i_vecText[-3].anim.glow.set(alpha=0),
            i_prodXText[-1].anim.glow.set(alpha=0),
            duration=0.25,
        )
        self.forward(0.75)
        self.play(
            i_vecText[-2].anim.glow.set(**glowCfg),
            i_prodYText[-1].anim.glow.set(**glowCfg),
            duration=0.25,
        )
        self.forward(0.75)
        self.play(
            i_vecText[-2].anim.glow.set(alpha=0),
            i_prodYText[-1].anim.glow.set(alpha=0),
            duration=0.25,
        )
        self.forward(2)

        i_skewedCoord = (
            NumberPlane(
                x_range=(-10, 15, 1),
                y_range=(-3, 6, 1),
                background_line_style=dict(stroke_alpha=0.75),
                # x_axis_config=dict(unit_size=0.75),
                # y_axis_config=dict(unit_size=0.75),
                depth=2,
            )
            .points.shift(i_coord.get_origin())
            .r
        )
        self.show(i_skewedCoord)
        self.hide(i_coord)

        trVec = mat @ vec
        i_trUVecText = addBgRect(
            TypstMath(
                f"""
                boldup(u) = vec({mat[0, 0]}, {mat[1, 0]}) quad
                boldup(v) = vec({mat[0, 1]}, {mat[1, 1]})
                """,
                preamble=preamble,
            )
            .points.to_border(UL, buff=0.25)
            .r
        )
        i_trUDotText = addBgRect(
            TypstMath(
                f"angle.l boldup(u), boldup(v) angle.r = {np.dot(mat[:, 0], mat[:, 1]):.0f}",
                preamble=preamble,
            )
            .points.next_to(i_trUVecText, DOWN, buff=0.25, aligned_edge=LEFT)
            .r
        )

        self.prepare(
            FadeOut(Group(i_rightAngle, i_prodXText, i_prodYText), duration=0.5),
            Transform(i_uVecText, i_trUVecText, duration=1),
            Transform(i_uDotText, i_trUDotText, duration=1),
        )

        self.play(
            i_skewedCoord.anim.points.apply_matrix(
                mat, about_point=i_coord.get_origin()
            ),
            i_ux.anim.points.put_start_and_end_on(
                i_coord.get_origin(), i_coord.c2p(*mat[:, 0])
            ),
            i_uy.anim.points.put_start_and_end_on(
                i_coord.get_origin(), i_coord.c2p(*mat[:, 1])
            ),
            i_vec.anim.points.put_start_and_end_on(
                i_coord.get_origin(), i_coord.c2p(*trVec)
            ),
            i_projX.anim.points.put_start_and_end_on(
                i_coord.get_origin(), i_coord.c2p(*(mat[:, 0] * vec[0]))
            ),
            i_projY.anim.points.put_start_and_end_on(
                i_coord.get_origin(), i_coord.c2p(*(mat[:, 1] * vec[1]))
            ),
            i_lineToX.anim.points.put_start_and_end_on(
                i_coord.c2p(*trVec), i_coord.c2p(*(mat[:, 0] * vec[0]))
            ),
            i_lineToY.anim.points.put_start_and_end_on(
                i_coord.c2p(*trVec), i_coord.c2p(*(mat[:, 1] * vec[1]))
            ),
            i_uxText.anim.points.next_to(i_coord.c2p(*mat[:, 0]), DOWN, buff=0.2),
            i_uyText.anim.points.next_to(i_coord.c2p(*mat[:, 1]), UL, buff=0.1),
            i_vecText.anim.points.next_to(i_coord.c2p(*trVec), RIGHT, buff=0.1),
            duration=3,
        )
        self.play(
            Group(
                i_skewedCoord,
                i_ux,
                i_uy,
                i_vec,
                i_projX,
                i_projY,
                i_lineToX,
                i_lineToY,
                i_vecText,
                i_uxText,
                i_uyText,
            ).anim.points.shift((-2.5, -0.5, 0)),
            duration=0.75,
        )
        i_coord.points.shift((-2.5, -0.5, 0))
        self.forward(0.5)
        self.play(
            *(
                ShowPassingFlash(i_.copy().set(**passingFlashCfg), time_width=2)
                for i_ in i_skewedCoord.get_axes()
            ),
            lag_ratio=0.25,
            duration=2,
        )
        self.forward(0.5)

        trProjU = np.dot(trVec, mat[:, 0]) / np.dot(mat[:, 0], mat[:, 0])
        trProjV = np.dot(trVec, mat[:, 1]) / np.dot(mat[:, 1], mat[:, 1])

        i_trProjXText = (
            TypstMath(
                f"""
                (angle.l boldup(u), boldup(alpha) angle.r) / abs(boldup(u))^2
                = {trProjU:.0f}
                != boldup(alpha)_boldup(u) = {vec[0]}
                """,
                preamble=preamble,
            )
            .points.next_to(i_coord.c2p(*(mat[:, 0] * trProjU)), DOWN, buff=(0, 0.2, 0))
            .r
        )
        i_trProjYText = (
            addBgRect(
                TypstMath(
                    f"""
                    (angle.l boldup(v), boldup(alpha) angle.r) / abs(boldup(v))^2
                    = {trProjV:.0f}
                    != boldup(alpha)_boldup(v) = {vec[1]}
                    """,
                    preamble=preamble,
                )
                .points.next_to(ORIGIN, UL, buff=(0, 0.2, 0))
                .r
            )
            .points.rotate(PI / 4, about_point=ORIGIN)
            .shift(i_coord.c2p(*(mat[:, 1] * trProjV)))
            .r
        )
        addBgRect(i_trProjXText)

        print(trProjU, trProjV)

        i_trProjX = Line(
            i_coord.get_origin(),
            i_coord.c2p(*(mat[:, 0] * trProjU)),
            stroke_color=RED,
            stroke_alpha=0.75,
        )
        i_trProjY = Line(
            i_coord.get_origin(),
            i_coord.c2p(*(mat[:, 1] * trProjV)),
            stroke_color=GREEN,
            stroke_alpha=0.75,
        )
        i_trLineToX = Line(
            i_coord.c2p(*trVec), i_coord.c2p(*(mat[:, 0] * trProjU)), **lineToAxisCfg
        )
        i_trLineToY = Line(
            i_coord.c2p(*trVec), i_coord.c2p(*(mat[:, 1] * trProjV)), **lineToAxisCfg
        )

        self.play(Uncreate(i_projX), Uncreate(i_projY))
        self.play(
            Create(i_trLineToX),
            Transform(i_vec, i_trProjX, root_only=True, hide_src=False),
        )
        self.play(Write(i_trProjXText), duration=0.75)
        self.forward(0.5)
        self.play(
            Create(i_trLineToY),
            Transform(i_vec, i_trProjY, root_only=True, hide_src=False),
        )
        self.play(Write(i_trProjYText), duration=0.75)
        self.forward(0.5)

        glowCfg = frozendict(size=0.5, alpha=0.5, color=PURE_RED)

        for _ in range(2):
            i_vecText[-3].glow.set(**glowCfg)
            i_trProjXText[-6].glow.set(**glowCfg)
            i_trProjXText[-1].glow.set(**glowCfg)
            self.forward(0.25)
            i_vecText[-3].glow.set(alpha=0)
            i_trProjXText[-6].glow.set(alpha=0)
            i_trProjXText[-1].glow.set(alpha=0)
            self.forward(0.25)
        self.forward(0.5)
        for _ in range(2):
            i_vecText[-2].glow.set(**glowCfg)
            i_trProjYText[-6].glow.set(**glowCfg)
            i_trProjYText[-1].glow.set(**glowCfg)
            self.forward(0.25)
            i_vecText[-2].glow.set(alpha=0)
            i_trProjYText[-6].glow.set(alpha=0)
            i_trProjYText[-1].glow.set(alpha=0)
            self.forward(0.25)

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
#############################################################################
