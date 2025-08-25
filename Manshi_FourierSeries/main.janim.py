from fractions import Fraction as Q

from janim.imports import *
import numpy as np
import pyrsistent as pyr

with reloads():
    from common import *


def circle3pData(p1: Vect, p2: Vect, p3: Vect) -> tuple[Vect, float]:
    x1, y1, _ = p1
    x2, y2, _ = p2
    x3, y3, _ = p3

    a, b, c = 2 * (x2 - x1), 2 * (y2 - y1), (x2**2 - x1**2) + (y2**2 - y1**2)
    d, e, f = 2 * (x3 - x2), 2 * (y3 - y2), (x3**2 - x2**2) + (y3**2 - y2**2)

    delta = a * e - b * d

    h = (c * e - b * f) / delta
    k = (a * f - c * d) / delta
    r = np.sqrt((x1 - h) ** 2 + (y1 - k) ** 2)

    return np.array((h, k, 0)), r


def circle3p(p1: Vect, p2: Vect, p3: Vect) -> Circle:
    center, radius = circle3pData(p1, p2, p3)
    return Circle(radius).points.shift(center).r


def perpPoint(p1: Vect, p2: Vect, p3: Vect) -> Vect:
    p1, p2, p3 = np.array(p1), np.array(p2), np.array(p3)
    v23 = p3 - p2
    v21 = p1 - p2
    dot = np.dot(v23, v21)
    mag = np.dot(v23, v23)
    t = dot / mag
    p4 = p2 + t * v23
    return p4


def nvec2d(v: Vect) -> Vect:
    x, y, *_ = v
    return np.array((-y, x, 0))


def unvec2d(v: Vect) -> Vect:
    nvec = nvec2d(v)
    return nvec / np.linalg.norm(nvec)


class MarkedText(Text, MarkedItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mark.set_points((ORIGIN,))


class TL_Geometry_Pythagorean(Timeline):
    def construct(self):
        self.camera.points.shift((3, 0.1, 0))
        theta = PI / 3
        r = 1.25

        alpha = theta / 2
        c = 2 * r
        a = c * np.sin(alpha)
        b = c * np.cos(alpha)

        trianglePoints = (
            (r, 0, 0),
            (r * np.cos(theta), r * np.sin(theta), 0),
            (-r, 0, 0),
        )
        points1 = (
            trianglePoints[0],
            (r, -2 * r, 0),
            (-r, -2 * r, 0),
            trianglePoints[-1],
        )
        points2 = (
            trianglePoints[0],
            (r + a * np.cos(alpha), a * np.sin(alpha), 0),
            (
                r + a * np.cos(alpha) - a * np.sin(alpha),
                a * np.sin(alpha) + a * np.cos(alpha),
                0,
            ),
            trianglePoints[1],
        )
        points3 = (
            trianglePoints[-1],
            (-r - b * np.sin(alpha), b * np.cos(alpha), 0),
            (
                -r - b * np.sin(alpha) + b * np.cos(alpha),
                b * np.cos(alpha) + b * np.sin(alpha),
                0,
            ),
            trianglePoints[1],
        )

        i_triangle = Polygon(*trianglePoints).radius.set(0.015).r
        i_square1 = Polyline(*points1).radius.set(0.015).r
        i_square2 = Polyline(*points2).radius.set(0.015).r
        i_square3 = Polyline(*points3).radius.set(0.015).r

        i_square1_1 = Polygon(*points1).stroke.set(GREEN).r.fill.set(GREEN, alpha=0.5).r
        i_square2_1 = (
            Polygon(*points2)
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r.points.reverse()
            .r
        )
        i_square3_1 = Polygon(*points3).stroke.set(RED).r.fill.set(RED, alpha=0.5).r

        i_line1 = Line(points1[0], points3[1]).radius.set(0.015).r
        i_line2 = Line(points1[2], points3[-1]).radius.set(0.015).r
        i_line3 = Line(points1[1], points2[-1]).radius.set(0.015).r
        i_line4 = Line(points1[-1], points2[1]).radius.set(0.015).r
        i_line5 = (
            Line(trianglePoints[1], (r * np.cos(theta), -2 * r, 0)).radius.set(0.015).r
        )

        i_triangle1_1 = (
            Polygon(trianglePoints[1], points3[1], trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )
        i_triangle1_2 = (
            Polygon(trianglePoints[0], points3[1], trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )
        i_triangle1_3 = (
            Polygon(points1[2], (r * np.cos(theta), 0, 0), trianglePoints[-1])
            .stroke.set(RED)
            .r.fill.set(RED, alpha=0.5)
            .r
        )

        i_triangle2_1 = (
            Polygon(trianglePoints[0], points2[1], points2[-1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )
        i_triangle2_2 = (
            Polygon(trianglePoints[0], points2[1], trianglePoints[-1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )
        i_triangle2_3 = (
            Polygon(trianglePoints[0], (r * np.cos(theta), 0, 0), points1[1])
            .stroke.set(BLUE)
            .r.fill.set(BLUE, alpha=0.5)
            .r
        )

        self.play(Create(i_triangle, auto_close_path=False), duration=0.5)
        self.play(
            AnimGroup(
                *(
                    Create(i_, auto_close_path=False, duration=0.5)
                    for i_ in (
                        i_square1,
                        i_square2,
                        i_square3,
                        i_line1,
                        i_line2,
                        i_line3,
                        i_line4,
                        i_line5,
                    )
                ),
                lag_ratio=0.25,
            )
        )
        self.play(FadeIn(i_triangle1_1), duration=0.5)
        self.play(Transform(i_triangle1_1, i_triangle1_2))
        self.play(
            Rotate(i_triangle1_2, -PI / 2, about_point=trianglePoints[-1]),
            duration=0.75,
        )

        self.play(FadeIn(i_triangle2_1), duration=0.5)
        self.play(Transform(i_triangle2_1, i_triangle2_2))
        self.play(
            Rotate(i_triangle2_2, PI / 2, about_point=trianglePoints[0]),
            duration=0.75,
        )

        self.play(
            Transform(i_triangle1_2, i_triangle1_3),
            Transform(i_triangle2_2, i_triangle2_3),
        )
        self.play(FadeOut(i_triangle1_3), FadeOut(i_triangle2_3), duration=0.5)
        self.play(FadeIn(i_square2_1), FadeIn(i_square3_1), duration=0.5)
        self.play(
            Transform(i_square2_1, i_square1_1, hide_src=False),
            Transform(i_square3_1, i_square1_1, hide_src=False),
            duration=1.5,
        )
        self.forward_to(12)


class TL_Geometry_NinePointCircle(Timeline):
    def construct(self):
        self.camera.points.shift((-2.25, -0.1, 0))
        trianglePoints = np.array(
            (
                (-2, -2.5, 0),
                (3.5, -2, 0),
                (0, 2, 0),
            )
        )
        midpoints = (trianglePoints + np.roll(trianglePoints, 1, axis=0)) / 2
        perpPoints = np.array(
            [
                perpPoint(*np.roll(trianglePoints, -i, axis=0))
                for i in range(len(trianglePoints))
            ]
        )
        center, radius = circle3pData(*midpoints)
        perpMidpoints = -midpoints + 2 * center

        i_triangle = Polygon(*trianglePoints).radius.set(0.015).r
        i_midTriangle = Polygon(*midpoints).radius.set(0.015).r
        i_circ = Circle(radius).points.shift(center).r.stroke.set(RED).r
        i_perpLines = Group(
            *(
                Line(p1, p2).radius.set(0.015).r
                for p1, p2 in zip(trianglePoints, perpPoints)
            )
        )
        i_diameters = Group(
            *(
                Line(p1, p2).radius.set(0.015).r
                for p1, p2 in zip(midpoints, perpMidpoints)
            )
        )
        i_center = Dot(center, radius=0.05)

        self.play(Create(i_triangle, auto_close_path=False), duration=1)
        self.play(
            AnimGroup(
                Create(i_midTriangle, auto_close_path=False),
                Create(i_circ, auto_close_path=False),
                lag_ratio=0.25,
            )
        )
        self.play(Create(i_perpLines))

        self.play(
            AnimGroup(
                *(Flash(point) for point in np.concat((midpoints, perpPoints), axis=0)),
                lag_ratio=0.5,
            )
        )
        self.play(FadeIn(i_center))
        self.play(Create(i_diameters))
        self.play(
            AnimGroup(
                *(Flash(point) for point in perpMidpoints),
                lag_ratio=0.5,
            )
        )

        self.forward_to(12)


class TL_Geometry(Timeline):
    def construct(self):
        tl1 = TL_Geometry_Pythagorean().build().to_item(keep_last_frame=True).show()
        tl2 = TL_Geometry_NinePointCircle().build().to_item(keep_last_frame=True).show()
        self.forward_to(max(tl1.end, tl2.end))


class TL_Polynomial(Timeline):
    CONFIG = config

    def construct(self):
        poly = np.polynomial.Polynomial((8, -2, -9, 2, 1))

        i_coord = (
            NumberPlane(
                x_range=(-8, 8, 1),
                y_range=(-64, 98, 16),
                depth=3,
                y_axis_config=dict(unit_size=1 / 16),
                background_line_style={"stroke_alpha": 0.75},
            )
            .points.shift((0.75, -0.25, 0))
            .r
        )
        i_graph = i_coord.get_graph(lambda _: 0, color=RED, depth=2)
        self.play(Create(i_coord))
        self.forward(1)

        i_diagrams = (
            Group(
                *(
                    PolyDiagram(
                        i,
                        x_extent=3,
                        y_extent=2.5,
                        x_axis_config=dict(unit_size=0.4),
                        y_axis_config=dict(unit_size=0.4),
                        axis_config=dict(tick_size=0.05),
                    )
                    for i in range(len(poly))
                )
            )
            .points.arrange_in_grid(n_rows=1, h_buff=0.25)
            .to_border(UP)
            .shift(UP * 0.25)
            .r
        )

        def createCoefText(coef: float, diagram: PolyDiagram) -> Text:
            i_text = (
                Text(
                    f"{coef:.2f}".replace("-", "\u2212"),
                    depth=-4,
                    stroke_alpha=1,
                    stroke_color=WHITE,
                    stroke_radius=0.005,
                )
                .points.scale(0.8)
                .next_to(diagram.points.box.get(DR), UL, buff=0.15)
                .r
            )
            i_text.add(
                SurroundingRect(
                    i_text,
                    buff=0.1,
                    stroke_alpha=0,
                    stroke_color=BLACK,
                    fill_color=BLACK,
                    fill_alpha=0.5,
                    depth=-3,
                )
            )
            return i_text

        i_coefTexts = Group(
            *(createCoefText(poly.coef[i], i_diagrams[i]) for i in range(len(poly)))
        )
        self.play(
            AnimGroup(*(FadeIn(i_) for i_ in i_diagrams), lag_ratio=0.25, duration=1)
        )
        self.forward(1)
        self.play(Create(i_graph))

        def createPolyGraphUpdaterFn(deg):
            def updaterFn(params: UpdaterParams) -> ParametricCurve:
                t = params.alpha
                interpPoly = np.polynomial.Polynomial(
                    np.append(poly.coef[:deg], poly.coef[deg] * t)
                )
                return i_coord.get_graph(interpPoly, color=RED, depth=2)

            return updaterFn

        def createCoefTextUpdateFn(deg):
            def updaterFn(params: UpdaterParams) -> Text:
                t = params.alpha
                coef = poly.coef[deg] * t
                return createCoefText(coef, i_diagrams[deg])

            return updaterFn

        for i in range(len(poly)):
            self.play(
                ItemUpdater(i_graph, createPolyGraphUpdaterFn(i)),
                ItemUpdater(i_coefTexts[i], createCoefTextUpdateFn(i)),
                duration=2,
            )
        self.forward(2)


class TL_Talor_Diagram(Timeline):
    _defaultPauses = pyr.m(start=0, beforeShowPoly=1)

    def __init__(
        self,
        coefsFactory=lambda: it.cycle((1, -1)),
        resultFn=lambda x: 1 / (x + 1),
        resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
        maxDeg=7,
        cropRadius=(2, 1.5),
        coordShift=(-0.5, -1),
        pauses=_defaultPauses,
        showResultFirst=True,
        *args,
        **kwargs,
    ):
        self._coefsFactory = coefsFactory
        self._resultFn = resultFn
        self._maxDeg = maxDeg
        self._resultGraphConfig = resultGraphConfig
        self._cropRadius = cropRadius
        self._coordShift = coordShift
        self._pauses = dict(self._defaultPauses)
        self._pauses.update(pauses)
        self.showResultFirst = showResultFirst
        super().__init__(*args, **kwargs)

    def construct(self):
        coefsFactory = self._coefsFactory
        resultFn = self._resultFn
        maxDeg = self._maxDeg
        cropRadius = self._cropRadius
        coordShift = self._coordShift
        pauses = self._pauses
        showResultFirst = self.showResultFirst

        i_coord = (
            Axes(num_sampled_graph_points_per_tick=5, axis_config=dict(tick_size=0.05))
            .points.shift((*coordShift, 0))
            .r
        )
        i_cropRect = Rect(
            *np.array(cropRadius) * 2, stroke_radius=0.03, stroke_color=WHITE
        )
        self.forward(pauses["start"])
        self.play(FadeIn(Group(i_cropRect, i_coord)))

        polys = tuple(
            np.polynomial.Polynomial(tuple(it.islice(coefsFactory(), i + 1)))
            for i in range(maxDeg)
        )
        i_resultGraph = i_coord.get_graph(
            resultFn,
            stroke_color=WHITE,
            stroke_alpha=0.5,
            **self._resultGraphConfig,
        )
        polyXRange = (
            i_coord.p2c((-cropRadius[0], 0, 0))[0],
            i_coord.p2c((cropRadius[0], 0, 0))[0],
        )
        i_polyGraph = i_coord.get_graph(polys[0], stroke_color=RED, x_range=polyXRange)

        if showResultFirst:
            self.play(Create(i_resultGraph), duration=1)
            self.forward(pauses["beforeShowPoly"])

        self.play(Create(i_polyGraph), duration=1)
        self.forward(0.5)
        for poly in polys[1:]:
            self.play(
                Transform(
                    i_polyGraph,
                    i_polyGraph := i_coord.get_graph(
                        poly, stroke_color=RED, x_range=polyXRange
                    ),
                ),
                duration=1,
            )
            self.forward(0.5)

        if not showResultFirst:
            self.play(Create(i_resultGraph), duration=1)

    @property
    def cropParams(self) -> tuple[float, float, float, float]:
        crx, cry = self._cropRadius
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        print(crx, cry, rx, ry)
        cropX, cropY = (rx - crx) / rx / 2, (ry - cry) / ry / 2
        print(cropX, cropY)
        return (cropX, cropY, cropX, cropY)


class TL_Talor(Timeline):
    def construct(self):
        i_poly1 = (
            PolynomialText(
                nameSymbol="1 / (x + 1)",
                terms=(RationalPolyTermOpt(1 - (i % 2) * 2, i) for i in range(7)),
                aligns=dict(coef=0),
            )
            .points.to_center()
            .shift(UP * 3)
            .r
        )
        i_poly2 = (
            PolynomialText(
                nameSymbol='integral_0^x ("d"t) / (t + 1)',
                terms=(
                    RationalPolyTermOpt(Q(1 - (i % 2) * 2, i + 1), i + 1)
                    for i in range(7)
                ),
                aligns=dict(coef=0),
            )
            .mark.set(i_poly1.mark.get())
            .r.points.shift(DOWN * 1.25)
            .r
        )
        i_convergenceRadius = (
            TypstMath("(-1 < x < 1)", fill_alpha=0.9)(VItem)
            .points.scale(0.8)
            .next_to(i_poly1, DOWN, buff=0.2, aligned_edge=RIGHT)
            .r
        )
        i_poly2NewSymbol = i_poly2.putSymbol(MarkedTypstMath("ln(x + 1)"))
        i_integral = (
            TypstMath('integral_0^x ("d" t) / (t + 1) = ln(x + 1)')
            .points.shift(DOWN * 0.25)
            .r
        )
        i_deriv = (
            TypstMath('("d") / ("d" x) ln(x + 1) = 1 / (x + 1)')
            .points.shift(DOWN * 2.25)
            .r
        )

        arrowRadius = 1.8
        i_arrows = (
            Group(
                Arrow(
                    LEFT * arrowRadius,
                    RIGHT * arrowRadius,
                    tip_kwargs=arrowCfg,
                    stroke_radius=0.015,
                ),
                Arrow(
                    RIGHT * arrowRadius,
                    LEFT * arrowRadius,
                    tip_kwargs=arrowCfg,
                    stroke_radius=0.015,
                ),
            )
            .points.arrange_in_grid(n_cols=1, v_buff=0.1)
            .shift(DOWN * 1.25)
            .r
        )

        tl1 = TL_Talor_Diagram(
            coefsFactory=lambda: it.cycle((1, -1)),
            resultFn=lambda x: 1 / (x + 1),
            resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
            maxDeg=7,
            coordShift=(-0.5, -1),
        )
        tl2 = TL_Talor_Diagram(
            coefsFactory=lambda: it.chain(
                (0,), ((-1 if i % 2 == 0 else 1) / i for i in it.count(1))
            ),
            resultFn=lambda x: np.log(x + 1),
            resultGraphConfig=pyr.m(x_range=(-0.8, 5)),
            maxDeg=8,
            coordShift=(-0.75, 0),
            pauses=dict(
                start=15.5,
            ),
            showResultFirst=False,
        )

        i_tl1 = tl1.build().to_item(keep_last_frame=True)
        i_tl1Clipped = TransformableFrameClip(
            i_tl1, offset=(-0.225, -0.15), clip=tl1.cropParams
        )

        i_tl2 = tl2.build().to_item(keep_last_frame=True)
        i_tl2Clipped = TransformableFrameClip(
            i_tl2, offset=(0.225, -0.15), clip=tl2.cropParams
        )

        self.show(i_tl1, i_tl2, i_tl1Clipped, i_tl2Clipped)

        self.play(
            Write(Group(*i_poly1.i_symbol, i_poly1.i_eq)), FadeIn(i_convergenceRadius)
        )
        self.forward(2)
        for i_sign, i_coef, i_term in zip(
            i_poly1.i_signs, i_poly1.i_coefs, i_poly1.i_terms
        ):
            self.play(*(FadeIn(i_) for i_ in (i_sign, i_coef, i_term)), duration=0.5)
            self.forward(1)
        self.play(FadeIn(i_poly1.i_ellipsis), FadeIn(i_poly1.i_signs[-1]))

        self.forward(1)
        self.play(
            *(FadeIn(i_poly2.i_symbol[i]) for i in (0, 2, 3, 4, 5)),
            Transform(i_poly1.i_symbol[2:5], i_poly2.i_symbol[6:9], hide_src=False),
            Transform(i_poly1.i_symbol[1], i_poly2.i_symbol[1], hide_src=False),
            Transform(i_poly1.i_eq, i_poly2.i_eq, hide_src=False),
            i_convergenceRadius.anim.points.next_to(
                i_poly2, DOWN, buff=0.2, aligned_edge=RIGHT
            ),
        )

        self.forward(1.5)
        for i_coef1, i_term1, i_sign1, i_coef2, i_term2, i_sign2 in zip(
            i_poly1.i_coefs,
            i_poly1.i_terms,
            i_poly1.i_signs,
            i_poly2.i_coefs,
            i_poly2.i_terms,
            i_poly2.i_signs,
        ):
            self.play(
                Transform(i_sign1, i_sign2, hide_src=False, flatten=True),
                Transform(i_coef1, i_coef2, hide_src=False, flatten=True),
                Transform(i_term1, i_term2, hide_src=False, flatten=True),
                duration=0.5,
            )
            self.forward(1)

        self.play(
            Transform(i_poly1.i_signs[-1], i_poly2.i_signs[-1], hide_src=False),
            Transform(i_poly1.i_ellipsis, i_poly2.i_ellipsis, hide_src=False),
            duration=0.5,
        )
        self.forward(1)
        self.play(Transform(i_poly2.i_symbol, i_poly2NewSymbol))
        self.forward(2)
        self.play(
            i_tl1Clipped.anim.clip.set(x_offset=-0.29),
            i_tl2Clipped.anim.clip.set(x_offset=0.29),
        )
        self.play(Write(i_integral, duration=1.5), Write(i_arrows[0], duration=1.5))
        self.forward(1)
        self.play(Write(i_deriv[::-1], duration=1.5), Write(i_arrows[1], duration=1.5))

        self.forward(2)


class TL_SineDeriv(Timeline):
    CONFIG = config

    def construct(self):
        def createCoord():
            i_coord = Axes(
                x_range=(-PI * 2, PI * 2, PI / 2),
                y_range=(-1.5, 1.5, 1),
                y_axis_config=dict(unit_size=0.75),
                x_axis_config=dict(unit_size=0.5),
                axis_config=dict(tick_size=0.05),
            )
            for i_axis in i_coord.get_axes():
                i_axis.ticks.set(stroke_radius=0.015)
            i_coord.add(
                SurroundingRect(i_coord, buff=0, color=WHITE, stroke_radius=0.015)
            )
            return i_coord

        i_coords = (
            Group(*(createCoord() for _ in range(4)))
            .points.arrange_in_grid(n_cols=2, h_buff=0.5, v_buff=1)
            .shift(UP * 0.75)
            .r
        )

        i_1 = i_coords[-1]
        i_2 = i_coords[-2]
        i_coords.remove(i_1, i_2)
        i_coords.add(i_1, i_2)

        functions = (np.sin, np.cos, lambda x: -np.sin(x), lambda x: -np.cos(x))
        colors = (RED, GREEN, BLUE, PINK)
        i_graphs = Group(
            *(i_.get_graph(f, color=c) for i_, f, c in zip(i_coords, functions, colors))
        )
        i_formulae = Group(
            TypstMath("sin x"),
            TypstMath("(sin x)' = cos x"),
            TypstMath("(sin x)''  = -sin x"),
            TypstMath("(sin x)''' = -cos x"),
        )

        for i_coord, i_formula in zip(i_coords, i_formulae):
            i_formula.points.move_to(i_coord.points.box.bottom).shift(DOWN * 0.5)

        i_lastCoord = None
        i_lastGraph = None

        def createTangentUpdaterFn(item: VItem) -> ItemUpdaterFn:
            def updaterFn(params: UpdaterParams) -> TangentLine:
                return TangentLine(item, params.alpha)

            return updaterFn

        def createDotUpdaterFn(i_graph: VItem) -> GroupUpdaterFn:
            def updaterFn(i_dot: Dot, params: UpdaterParams) -> None:
                i_dot.points.move_to(i_graph.points.pfp(params.alpha))

            return updaterFn

        def createLineUpdaterFn(
            i_graph: VItem, i_coord: CoordinateSystem
        ) -> GroupUpdaterFn:
            def updaterFn(i_line: Line, params: UpdaterParams) -> None:
                pointOnGraph = i_graph.points.pfp(params.alpha)
                x, _ = i_coord.p2c(pointOnGraph)
                pointOnAxis = i_coord.c2p(x, 0)
                i_line.points.put_start_and_end_on(pointOnGraph, pointOnAxis)

            return updaterFn

        for i_coord, i_graph, i_formula in zip(i_coords, i_graphs, i_formulae):
            if i_lastGraph is None:
                self.play(FadeIn(i_coord))
                self.play(Create(i_graph, duration=2), Write(i_formula))
            else:
                self.play(Transform(i_lastCoord, i_coord, hide_src=False, flatten=True))
                i_tangent = TangentLine(i_lastGraph, 0)
                i_dotOnGraph = Dot(i_graph.points.pfp(0), radius=0.06)
                i_dotOnLastGraph = Dot(i_lastGraph.points.pfp(0), radius=0.06)
                i_lineOnGraph = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
                i_lineOnLastGraph = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
                self.play(
                    *(
                        FadeIn(i_)
                        for i_ in (
                            i_tangent,
                            i_dotOnGraph,
                            i_dotOnLastGraph,
                            i_lineOnGraph,
                            i_lineOnLastGraph,
                        )
                    ),
                    duration=0.5,
                )
                self.play(
                    Create(i_graph),
                    ItemUpdater(i_tangent, createTangentUpdaterFn(i_lastGraph)),
                    GroupUpdater(i_dotOnGraph, createDotUpdaterFn(i_graph)),
                    GroupUpdater(i_dotOnLastGraph, createDotUpdaterFn(i_lastGraph)),
                    GroupUpdater(i_lineOnGraph, createLineUpdaterFn(i_graph, i_coord)),
                    GroupUpdater(
                        i_lineOnLastGraph, createLineUpdaterFn(i_lastGraph, i_lastCoord)
                    ),
                    duration=3,
                )
                self.play(
                    *(
                        FadeOut(i_)
                        for i_ in (
                            i_tangent,
                            i_dotOnGraph,
                            i_dotOnLastGraph,
                            i_lineOnGraph,
                            i_lineOnLastGraph,
                        )
                    ),
                    FadeIn(i_formula),
                    duration=0.5,
                )
            self.forward(1)
            i_lastGraph = i_graph
            i_lastCoord = i_coord

        i_sinGraph, i_negSinGraph = i_graphs[0], i_graphs[2]

        self.play(i_sinGraph.anim.glow.set(color=colors[0], alpha=0.5, size=0.75))
        self.forward(0.5)
        self.play(i_negSinGraph.anim.glow.set(color=colors[2], alpha=0.5, size=0.75))
        self.forward(0.5)
        self.play(ShowPassingFlashAround(i_formulae[2][-5], time_width=3, duration=2))
        self.forward(2)

        i_dotOnGraph1 = Dot(i_sinGraph.points.pfp(0), radius=0.06)
        i_dotOnGraph2 = Dot(i_negSinGraph.points.pfp(0), radius=0.06)
        i_lineOnGraph1 = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)
        i_lineOnGraph2 = Line(ORIGIN, ORIGIN, stroke_alpha=0.75)

        self.play(
            i_sinGraph.anim.glow.set(alpha=0).r.stroke.set(alpha=0.5),
            i_negSinGraph.anim.glow.set(alpha=0).r.stroke.set(alpha=0.5),
            *(
                FadeIn(i_)
                for i_ in (i_dotOnGraph1, i_dotOnGraph2, i_lineOnGraph1, i_lineOnGraph2)
            ),
            duration=0.5,
        )
        self.forward(0.5)

        i_sinGraphCp = i_sinGraph.copy().stroke.set(alpha=1).r
        i_negSinGraphCp = i_negSinGraph.copy().stroke.set(alpha=1).r

        self.play(
            Create(i_sinGraphCp),
            Create(i_negSinGraphCp),
            GroupUpdater(i_dotOnGraph1, createDotUpdaterFn(i_sinGraph)),
            GroupUpdater(i_dotOnGraph2, createDotUpdaterFn(i_negSinGraph)),
            GroupUpdater(i_lineOnGraph1, createLineUpdaterFn(i_sinGraph, i_coords[0])),
            GroupUpdater(
                i_lineOnGraph2, createLineUpdaterFn(i_negSinGraph, i_coords[2])
            ),
            duration=3,
        )
        self.forward(0.5)
        self.play(
            *(
                FadeOut(i_)
                for i_ in (i_dotOnGraph1, i_dotOnGraph2, i_lineOnGraph1, i_lineOnGraph2)
            ),
            duration=0.5,
        )
        # self.play(Transform(i_graphs[0], i_graphs[2], hide_src=False), duration=2)

        self.forward(2)


class TL_VecDotProduct(Timeline):
    CONFIG = config

    def construct(self) -> None:
        rx, ry = self.config_getter.frame_x_radius, self.config_getter.frame_y_radius
        i_coord = (
            NumberPlane(
                x_range=(-10, 10),
                y_range=(-6, 6),
                background_line_style=dict(stroke_alpha=0.75),
                x_axis_config=dict(unit_size=0.75),
                y_axis_config=dict(unit_size=0.75),
                depth=2,
            )
            .points.shift((-3, 0, 0))
            .r
        )

        i_dotProductFormula = (
            TypstMath(
                f"""
                angle.l
                text(arrow.tr, fill: #rgb("{RED}")),
                text(arrow.br, fill: #rgb("{GREEN}"))
                angle.r =
                """,
                depth=-1,
            )
            .points.to_border(UL)
            .r
        )
        i_dotProductFormula[1].set(
            stroke_alpha=1, stroke_radius=0.005, stroke_color=RED
        )
        i_dotProductFormula[3].set(
            stroke_alpha=1, stroke_radius=0.005, stroke_color=GREEN
        )

        def createDotProductValueText(value: float) -> Text:
            i_dotProductValue = (
                Text(f"{value:.2f}".replace("-", "\u2212"), depth=-1)
                .points.next_to(i_dotProductFormula, RIGHT, buff=0.2)
                .r
            )
            return i_dotProductValue

        i_dotProductValue = createDotProductValueText(0)
        i_bgRect = SurroundingRect(
            Group(i_dotProductFormula, i_dotProductValue),
            buff=0.2,
            fill_color=BLACK,
            fill_alpha=0.75,
            stroke_alpha=0,
            stroke_color=BLACK,
            stroke_radius=0,
            depth=1,
        )

        def createValueUpdaterFn(start1, end1, start2, end2):
            def updaterFn(params: UpdaterParams) -> None:
                t = params.alpha
                p1 = interpolate(start1, end1, t)
                p2 = interpolate(start2, end2, t)
                value = np.dot(p1, p2)
                return createDotProductValueText(value)

            return updaterFn

        def createVecUpdaterFn(start, end):
            start = np.array(start)
            end = np.array(end)

            def updaterFn(item: Arrow, params: UpdaterParams) -> None:
                t = params.alpha
                point = interpolate(start, end, t)
                item.points.put_start_and_end_on(coordOrigin, i_coord.c2p(*point))

            return updaterFn

        origin = np.array((0, 0))
        vec1 = np.array((2, 2))
        vec2 = np.array((4, -1))

        coordOrigin = i_coord.c2p(0, 0)
        i_vec1 = Arrow(coordOrigin, coordOrigin, buff=0, color=RED)
        i_vec2 = Arrow(coordOrigin, coordOrigin, buff=0, color=GREEN)

        def vecAnim(start1, end1, start2, end2):
            return AnimGroup(
                GroupUpdater(i_vec1, createVecUpdaterFn(start1, end1)),
                GroupUpdater(i_vec2, createVecUpdaterFn(start2, end2)),
                ItemUpdater(
                    i_dotProductValue, createValueUpdaterFn(start1, end1, start2, end2)
                ),
            )

        i_formulae = (
            TypstDoc(
                (DIR / "assets/typst/vec-dot-product-properties.typ").read_text(
                    encoding="utf-8"
                ),
                depth=-1,
            )
            .points.move_to((4.8, 0.25, 0))
            .r
        )
        i_formula1 = i_formulae[0:24]
        i_formula2 = i_formulae[24:38]
        i_formula3 = i_formulae[38:]

        i_formulaeBackground = Rect(
            (rx / 3, -ry - 0.25, 0),
            (rx + 0.25, ry + 0.25, 0),
            fill_color=BLACK,
            fill_alpha=0.75,
            stroke_color=WHITE,
            stroke_alpha=0.75,
            stroke_radius=0.01,
            depth=1,
        )

        self.play(Create(i_coord), FadeIn(i_formulaeBackground))

        self.play(
            FadeIn(i_bgRect, duration=1),
            Succession(
                Write(i_dotProductFormula, duration=0.75),
                FadeIn(i_dotProductValue, duration=0.25),
                duration=1,
            ),
        )

        self.show(i_vec2, i_vec1)
        self.play(vecAnim(origin, vec1, origin, vec2), duration=2)
        self.forward(2)

        # 双线性性

        def createMultText(vec, mult, direction=1) -> Text:
            direction = 1 if direction >= 0 else -1
            vec = np.array(vec)
            nvec = unvec2d(vec)
            x, y, *_ = vec
            dirAngle = np.atan2(y, x)
            i_multText = (
                Text(f"{mult:.2f}×".replace("-", "\u2212"), depth=-1)
                .points.move_to(coordOrigin)
                .r
            )
            (
                i_multText.add(
                    SurroundingRect(
                        i_multText,
                        buff=0.1,
                        stroke_color=BLACK,
                        stroke_radius=0,
                        fill_alpha=0.75,
                        fill_color=BLACK,
                    )
                )
                .points.rotate(dirAngle, about_point=coordOrigin)
                .move_to(i_coord.c2p(*(vec * (mult / 2))))
                .shift(nvec * (0.75 * direction))
            )
            return i_multText

        def animateVecScale(idx, mults=(2,)):
            direction = 1 if idx == 0 else -1
            i_vec = i_vec1 if idx == 0 else i_vec2
            i_vecCp = i_vec.copy().set(fill_alpha=0.75, stroke_alpha=0.75)
            vec = vec1 if idx == 0 else vec2

            unvec = unvec2d(vec)
            i_brace = Brace(i_vec, unvec * direction)
            i_multText = createMultText(vec, 1, direction)
            self.play(
                *(FadeIn(i_) for i_ in (i_brace, i_multText, i_vecCp)), duration=0.5
            )
            self.play(
                ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=2)
            )
            self.forward(0.5)
            for lastMult, mult in it.pairwise(it.chain((1,), mults, (1,))):
                self.play(
                    vecAnim(vec1 * lastMult, vec1 * mult, vec2, vec2)
                    if idx == 0
                    else vecAnim(vec1, vec1, vec2 * lastMult, vec2 * mult),
                    DataUpdater(
                        i_brace, lambda i_, _: i_.points.match(i_vec.current())
                    ),
                    ItemUpdater(
                        i_multText,
                        lambda params, lastMult=lastMult, mult=mult: createMultText(
                            vec, interpolate(lastMult, mult, params.alpha), direction
                        ),
                    ),
                    duration=0.75,
                )
                self.play(
                    ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1)
                )
                self.forward(0.5)
            self.play(
                *(FadeOut(i_) for i_ in (i_brace, i_multText, i_vecCp)), duration=0.5
            )

        animateVecScale(0, (2,))
        self.forward(1)
        animateVecScale(1, (1.5,))
        self.play(Write(i_formula1), duration=1)
        self.forward(2)

        # 对称性
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=2))
        self.forward(0.5)
        self.play(vecAnim(vec1, vec2, vec2, vec1), duration=2)
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1))
        self.forward(0.5)
        self.play(vecAnim(vec2, vec1, vec1, vec2), duration=2)
        self.play(ShowPassingFlashAround(i_dotProductValue, time_width=3, duration=1))
        self.play(Write(i_formula2), duration=1)
        self.forward(2)

        # 正定性
        self.play(vecAnim(vec1, vec1, vec2, vec1))
        self.play(
            i_vec2.anim.set(stroke_alpha=0, fill_alpha=0),
            i_dotProductFormula[3]
            .anim.points.rotate(PI / 2)
            .r.fill.set(color=RED)
            .r.stroke.set(color=RED),
            duration=0.5,
        )
        self.forward(1)

        targetPoints = np.array(((4, -3), (-4, -2), (-3, 3), vec1))
        for lastP, p in it.pairwise(it.chain((vec1,), targetPoints)):
            self.play(vecAnim(lastP, p, lastP, p), duration=1)
            self.forward(0.5)

        self.play(vecAnim(p, origin, p, origin), duration=2)
        self.play(Write(i_formula3), duration=1)

        self.forward(2)


class TL_Test(Timeline):
    def construct(self):
        i_arrow = Arrow(ORIGIN, ORIGIN, buff=0)
        self.play(Create(i_arrow))
        i_arrow.points.put_start_and_end_on(ORIGIN, np.array((3, 1, 0)))
        self.forward(2)


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
