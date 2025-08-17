from janim.imports import *

with reloads():
    from common import *


class TL_Coord(Timeline):
    CONFIG = config

    def construct(self):
        u = 0.75
        coords = np.array(
            (
                (4, 3),
                (-3, 4),
                (-5, -2),
                (5, -3),
            )
        )
        startCoord = endCoord = coords[0]
        i_coord = NumberPlane(
            x_range=(-10, 10, 1),
            y_range=(-7, 7, 1),
            background_line_style={"stroke_alpha": 0.75},
            unit_size=u,
        )
        originPoint = i_coord.c2p(0, 0)
        startPoint = i_coord.c2p(*startCoord)
        x0, y0, _ = originPoint
        x1, y1, _ = startPoint

        i_dot = Dot(startPoint, radius=0.05).depth.set(-1).r
        i_lineToX = Line(startPoint, (x1, y0, 0)).stroke.set(color=RED).r
        i_lineToY = Line(startPoint, (x0, y1, 0)).stroke.set(color=GREEN).r

        def createCoordText(coord):
            x, y = coord
            return (
                Text(f"({x:.2f}, {y:.2f})")
                .points.next_to(i_coord.c2p(*coord), aligned_edge=DL)
                .r
            )

        i_coordText = createCoordText(startCoord)

        self.play(Create(i_coord), duration=2)
        self.forward(1)
        self.play(FadeIn(i_dot))
        self.play(Write(i_coordText))
        self.play(Create(i_lineToX))
        self.play(Create(i_lineToY))

        def getUpdaterFns(startCoord, endCoord):
            def updateDot(i_: Dot, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                i_.points.move_to(i_coord.c2p(*coord))

            def updateLineToX(i_: Line, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                x, y = coord
                i_.points.put_start_and_end_on(i_coord.c2p(x, y), i_coord.c2p(x, 0))

            def updateLineToY(i_: Line, params: UpdaterParams) -> None:
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                x, y = coord
                i_.points.put_start_and_end_on(i_coord.c2p(x, y), i_coord.c2p(0, y))

            def updateCoordText(params: UpdaterParams):
                t = params.alpha
                coord = startCoord * (1 - t) + endCoord * t
                return createCoordText(coord)

            return updateDot, updateLineToX, updateLineToY, updateCoordText

        def animatePointMove(startCoord, endCoord):
            updaterFns = getUpdaterFns(startCoord, endCoord)
            self.play(
                i_dot.anim.points.move_to(i_coord.c2p(*coord)),
                *(
                    GroupUpdater(i_, updater)
                    for i_, updater in zip(
                        (i_dot, i_lineToX, i_lineToY),
                        updaterFns[:-1],
                    )
                ),
                ItemUpdater(i_coordText, updaterFns[-1]),
                duration=1.5,
            )

        for coord in coords[1:]:
            endCoord = coord
            animatePointMove(startCoord, endCoord)
            startCoord = endCoord

        endCoord = coords[0]
        animatePointMove(startCoord, endCoord)

        self.forward(2)

        i_ux = Line(originPoint, i_coord.c2p(1, 0)).stroke.set(color=RED).r
        i_uy = Line(originPoint, i_coord.c2p(0, 1)).stroke.set(color=GREEN).r

        i_ux.add_tip(fill_color=RED, **arrowConfig)
        i_uy.add_tip(fill_color=GREEN, **arrowConfig)

        self.forward(1)
        self.play(
            Uncreate(i_lineToX),
            Uncreate(i_lineToY),
            FadeOut(i_coordText),
            FadeOut(i_dot),
        )

        startCoord = endCoord

        i_lineOnX = (
            Line(originPoint, i_coord.c2p(startCoord[0], 0)).stroke.set(color=RED).r
        )
        i_lineOnY = (
            Line(originPoint, i_coord.c2p(0, startCoord[1])).stroke.set(color=GREEN).r
        )
        i_tickOnX = Line(ORIGIN, UP * 0.15).points.move_to(originPoint).r
        i_tickOnY = Line(ORIGIN, RIGHT * 0.15).points.move_to(originPoint).r
        i_xText = Text("0.00").points.next_to(originPoint, DOWN).r
        i_yText = Text("0.00").points.next_to(originPoint, LEFT).r

        def updateLineOnX(i_: Line, params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            i_.points.put_start_and_end_on(originPoint, i_coord.c2p(endX * t, 0))

        def updateTickOnX(i_: Line, params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            i_.points.move_to(i_coord.c2p(endX * t, 0))

        def updateLineOnY(i_: Line, params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            i_.points.put_start_and_end_on(originPoint, i_coord.c2p(0, endY * t))

        def updateTickOnY(i_: Line, params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            i_.points.move_to(i_coord.c2p(0, endY * t))

        def updateXText(params: UpdaterParams):
            t = params.alpha
            endX = startCoord[0]
            x = endX * t
            return Text(f"{x:.2f}").points.next_to(i_coord.c2p(x, 0), DOWN).r

        def updateYText(params: UpdaterParams):
            t = params.alpha
            endY = startCoord[1]
            y = endY * t
            return Text(f"{y:.2f}").points.next_to(i_coord.c2p(0, y), LEFT).r

        self.play(
            GroupUpdater(i_lineOnX, updateLineOnX),
            GroupUpdater(i_tickOnX, updateTickOnX),
            ItemUpdater(i_xText, updateXText),
            duration=1.5,
        )
        self.play(
            GroupUpdater(i_lineOnY, updateLineOnY),
            GroupUpdater(i_tickOnY, updateTickOnY),
            ItemUpdater(i_yText, updateYText),
            duration=1.5,
        )
        i_lineToX.points.reverse()
        i_lineToY.points.reverse()
        self.play(Create(i_lineToX), Create(i_lineToY))
        self.play(Write(i_coordText), FadeIn(i_dot))

        self.forward(1)
        self.play(Uncreate(i_lineOnX), Uncreate(i_lineOnY))
        self.play(Create(i_ux), duration=1)
        self.play(Create(i_uy), duration=1)

        i_perp = RightAngle(i_ux, i_uy, length=0.15)
        self.play(FadeIn(i_perp))

        self.forward(2)


if __name__ == "__main__":
    import subprocess

    subprocess.run(["janim", "run", __file__, "-i"])
