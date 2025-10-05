import pyagrum as gum

#creating net
bn = gum.BayesNet("Colorectal Cancer Treatment")

#creating variables
id_c = bn.add(gum.LabelizedVariable("c", "cloudy ?", 2))
id_s = bn.add(gum.LabelizedVariable("s", "sprinkler ?", 2))
id_r, id_w = [
    bn.add(name,2) for name in "srw"
]

#creating arcs
bn.addArc("c", "s")
for link in [(id_c, id_r), ("s", "w"), ("r", "w")]:
    bn.addArc(*link)

#creating probability tables
bn.cpt(id_c).fillWith([0.4, 0.6])

