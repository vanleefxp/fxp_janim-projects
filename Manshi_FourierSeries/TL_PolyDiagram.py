from janim.imports import *

with reloads():
    from common import *


class TL_PolyDiagrams(Timeline):
    def construct(self):
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        # bgColor = self.config_getter.background_color
        # print(bgColor)
        blockHeight = 0.75
        i_block = Rect(
            (-rx, -ry, 0),
            (rx, -ry + blockHeight, 0),
            fill_color=BLACK,
            fill_alpha=1,
            stroke_color=BLACK,
            stroke_alpha=0,
            stroke_radius=0,
            depth=-114514,
        )
        self.show(i_block)
        i_polyDiagrams = (
            Group(
                *(
                    PolyDiagram(
                        degree=i,
                        axis_config=dict(tick_size=0.05),
                        x_axis_config=dict(unit_size=0.5),
                        y_axis_config=dict(unit_size=0.5),
                    )
                    for i in range(36)
                )
            )
            .points.arrange_in_grid(n_cols=4, h_buff=0.5, v_buff=0.5)
            .to_border(UP, buff=0.25)
            .r
        )
        # self.show(i_polyDiagrams)
        n_anim = 12
        self.play(
            AnimGroup(
                *(GrowFromPoint(i_, ORIGIN) for i_ in i_polyDiagrams[:n_anim]),
                lag_ratio=0.1,
                duration=2.5,
            )
        )
        self.show(*i_polyDiagrams[n_anim:])
        self.forward(2)
        self.play(i_polyDiagrams.anim.points.next_to(i_block, UP, buff=0.1), duration=4)
        self.forward(1)
        self.play(i_polyDiagrams.anim.points.to_border(UP, buff=0.25), duration=2)
        self.play(
            AnimGroup(
                *(FadeOut(i_) for i_ in i_polyDiagrams[:n_anim]),
                lag_ratio=0.1,
                duration=1,
            )
        )
        self.hide(*i_polyDiagrams[n_anim:])


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
