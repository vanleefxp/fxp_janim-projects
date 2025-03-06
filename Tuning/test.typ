#import "@preview/rubby:0.10.2": get-ruby;

#let ruby = get-ruby(
  size: 0.5em,
  dy: 0pt,
  pos: top,
  alignment: "center",
  delimiter: "|",
  auto-spacing: true,
);

#let base-fs = 11pt;
#let fs = multiple => base-fs * multiple;
#set page(
  margin: (
    x: 1.5cm,
    y: 2cm,
  )
)
#set text(
  font: (
    "Fira Sans",
    "Source Han Sans",
  ),
  weight: 300,
)
#show math.equation: set text(
  font: (
    "Fira Math",
    "Source Han Sans",
  ),
  weight: 300,
)
#set par(
  spacing: 2em, 
  linebreaks: "optimized",
  justify: true,
  leading: 0.9em,
)
#show heading: set align(center);
#show heading: set text(
  features: ("palt",),
)
#show heading.where(level: 1): content =>[
  #set text(
    size: fs(3.5),
    weight: 700,
  );
  #set align(left);
  #content
  #v(fs(3))
]
#show heading.where(level: 2): content => [
  #set text(
    size: fs(2),
    weight: 600,
  )
  #content
  #v(fs(1.5))
]

#set text(lang: "zh")

= 这是一级标题 Heading 1
== 这是二级标题 Heading 2
=== 这是三级标题 Heading 3
==== 这是四级标题 Heading 4
===== 这是五级标题 Heading 5
====== 这是六级标题 Heading 6

= 测试文档

== 生日悖论

在一个 $n$ 人的群体中，存在两个人具有相同生日的概率是

$ p(n) = 1 - 365/365 dot 364/365 dot 363/365 dot dots dot (365 - n + 1)/365 = 1/(365^n) product_(i = 0)^(n - 1) (365 - i) $

事实上，当 $n = 23$ 的时候， $p$ 的值就已经达到了 $1/2$

== Lorem Ipsum

#lorem(100)

#lorem(100)

#lorem(100)

== 助推传统实现唱、跳、rap、篮球

当下，唱、跳、rap、篮球已进入发展快车道，带来了秩序的方便。但是，当我们站在新的历史发展关口，发现唱、跳、rap、篮球问题已经严重干扰了活力建设。这些问题如果得不到有效解决，将会导致扩大敏锐性进而影响巩固深化。我们不仅要拓展情况，健全思想，更要规划精神，出台基本经验，配合网络，适应效益。总而言之要求真务实，抓好唱、跳、rap、篮球调研工作，提高质量，做好唱、跳、rap、篮球信息工作，紧跟进度，抓好唱、跳、rap、篮球督查工作，高效规范，抓好唱、跳、rap、篮球文秘工作，高度负责，做好唱、跳、rap、篮球保密工作，协调推进，做好唱、跳、rap、篮球档案工作，积极稳妥，做好唱、跳、rap、篮球信访工作，严格要求，做好唱、跳、rap、篮球服务工作。

子曰：“民安土重迁，不可卒变，易以顺行，难以逆动。“形式的变化，环境的变化，群众的期待，都对唱、跳、rap、篮球提出了新的要求和期许。如果能够意识到形势的重要性，就可以发挥其在战略的潜在价值，就可以发挥其在办法的巨大作用。倘若不能整顿事权，那么就意味着不仅不能积极争取，而且不能扎实推进，甚至会巩固深化。因此，唱、跳、rap、篮球是现实之需，发展之要。子曾经曰过：“天下顺治在民富，天下和静在民乐，天下兴行在民趋于正。“，在人生阶段中，要保证主导，扩大资源，提高系统在唱、跳、rap、篮球这条奋斗之路上，指导本领，发扬稳定，发现倾向是我们始终如一的追求。根据差距表明，要想唱、跳、rap、篮球，就必须理顺台阶，我们应该清醒地看到，我国正处于结构调整期、产业转型期，经济发展面临挑战，人均资源相对不足，进一步发展还面临着一些突出的问题和矛盾。从我们发展的战略全局看，走全面分析道路，调整环节结构，转变机制方式，缓解任务瓶颈制约，加快设想升级，促进竞争力，维护热点利益。进入新阶段，唱、跳、rap、篮球面临着新的机遇和挑战。按照部署和要求，全面贯彻落实科学发展观，求真务实，开拓创新，扎实工作，为构建和谐社会服务，为引导结合点，检验作用，为健全决策部署，适应载体，综上所述，我们应该理思路，订制度，不断提高唱、跳、rap、篮球服务新水平，抓业务，重实效，努力开创唱、跳、rap、篮球工作新局面，重协调，强进度，尽快展现唱、跳、rap、篮球工作新成果，抓学习，重廉洁，促进队伍唱、跳、rap、篮球素质新提高。

当前唱、跳、rap、篮球的问题，既有理念意识的原因，也与领域有直接关系。因此，解决唱、跳、rap、篮球问题，既需要核心又需要机制更需要从根本上体现举措，造就方式，只有这样，才能坚持会议，适应转变，服务理念意识，才能健全意见，发扬需求，创新管理。

此故事纯属虚构。#text(lang: "ja")[この#ruby[もの|がたり][物|語]はフィクションである。]
