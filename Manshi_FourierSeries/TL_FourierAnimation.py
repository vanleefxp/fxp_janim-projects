from janim.imports import *
from common import *
from fourier_figure import FourierFigure

with reloads():
    from common import *
    from fourier_figure import FourierFigure


# Manim 经典案例：傅里叶级数动画
# 此处用 JAnim 来复现此经典动画
class TL_FourierSeriesAnim(Timeline):
    CONFIG = config

    def construct(self):
        coefs_file = DIR / "assets/data/fourier-coefs/like.npz"
        max_n = 40
        scale = 2.5
        startPoint = ORIGIN

        # 加载傅里叶级数图形
        # 限制最大向量对数
        fig: FourierFigure = FourierFigure(coefs_file)[:max_n]
        max_n = fig.max_n
        coefs = fig.coefs

        # 重新排序傅里叶系数
        coefs_alternating = fftalt(coefs)
        coefsabs_alternating = np.abs(coefs_alternating)

        # 生成目标图像
        fn_points = toPointsFn(fig)
        i_graph = ParametricCurve(
            lambda t: fn_points(t) * scale + startPoint, (0, 1, 0.001), color=BILI_PINK
        )

        # 生成向量、圆和点
        i_vecs = Group[Vector](
            *(
                Line(
                    ORIGIN,
                    ORIGIN,
                    buff=0,
                    stroke_radius=interpolate(0.01, 0.0025, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs = Group[Circle](
            *(
                Circle(
                    coefabs * scale,
                    stroke_radius=interpolate(0.01, 0.005, 1 - np.exp(-np.abs(i) / 5)),
                )
                for i, coefabs in zip(alternatingSignedInts(), coefsabs_alternating)
            )
        )
        i_points = Group[Dot](
            *(
                Dot(radius=interpolate(0.04, 0.01, 1 - np.exp(-np.abs(i) / 5)))
                for i in alternatingSignedInts(max_n)
            )
        )
        i_circs[0].set(stroke_alpha=0)
        i_graphingItems = Group(i_vecs, i_circs, i_points)
        i_startPoint = Dot(startPoint, radius=0.04)

        # 实时更新向量、圆和点的位置
        def updateVectors(t: float, items=i_graphingItems):
            i_vecs, i_circs, i_dots = items
            components = fig.components(t)
            components_cp = components.copy()
            components[0] = components_cp[0]
            components[1::2] = components_cp[1 : fig.max_n]
            components[2::2] = components_cp[-1 : -fig.max_n : -1]
            components_cumsum = np.cumsum(components)

            for (pc_prev, pc_next), i_vec, i_circ, i_dot in zip(
                it.pairwise(it.chain((0j,), components_cumsum)), i_vecs, i_circs, i_dots
            ):
                p_prev = complex2point(pc_prev * scale) + startPoint
                p_next = complex2point(pc_next * scale) + startPoint
                i_vec.points.set_start_and_end(p_prev, p_next)
                i_circ.points.move_to(p_prev)
                i_dot.points.move_to(p_next)

        updateVectors(0)
        self.forward(0.25)

        lag_ratio = 1 / 8
        ag1 = []
        ag2 = []
        ag3 = []
        for i_circ, i_line, i_point, coef in zip(
            i_circs,
            i_vecs,
            i_points,
            coefs_alternating,
        ):
            duration = np.sqrt(np.abs(coef))
            ag1.append(GrowFromCenter(i_circ, duration=duration))
            ag2.append(Create(i_line, duration=duration))
            ag3.append(
                GrowFromPoint(i_point, i_circ.points.self_box.center, duration=duration)
            )
        ag1 = AnimGroup(*ag1, lag_ratio=lag_ratio)
        ag2 = AnimGroup(*ag2, lag_ratio=lag_ratio)
        ag3 = AnimGroup(*ag3, lag_ratio=lag_ratio)
        self.play(FadeIn(i_startPoint), ag1, ag2, ag3, duration=3)

        self.forward(1)
        anim_duration = 15
        self.play(
            GroupUpdater(
                i_graphingItems,
                lambda item, params: updateVectors(params.alpha, item),
                duration=anim_duration,
                rate_func=linear,
            ),
            Create(
                i_graph, auto_close_path=False, duration=anim_duration, rate_func=linear
            ),
        )
        self.forward(1)

        lag_ratio = 1 / 8
        ag1 = []
        ag2 = []
        ag3 = []
        for i_circ, i_line, i_point, coef in zip(
            i_circs,
            i_vecs,
            i_points,
            coefs_alternating,
        ):
            duration = np.sqrt(np.abs(coef))
            ag1.append(ShrinkToCenter(i_circ, duration=duration))
            ag2.append(Uncreate(i_line, duration=duration))
            ag3.append(
                ShrinkToPoint(i_point, i_circ.points.self_box.center, duration=duration)
            )
        for ag in ag1, ag2, ag3:
            ag.reverse()
        ag1 = AnimGroup(*ag1, lag_ratio=lag_ratio)
        ag2 = AnimGroup(*ag2, lag_ratio=lag_ratio)
        ag3 = AnimGroup(*ag3, lag_ratio=lag_ratio)
        self.play(ag1, ag2, ag3, FadeOut(i_startPoint), duration=3)

        # print(coefs)
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
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
