"""Save Q446-Q450 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (446, "Linguistics", "How do children acquire language so rapidly?",
     "First words ~12 months, two-word phrases ~18-24 months, complex sentences by 3-4 years — from imperfect, fragmented input.\n\n"
     "**Poverty of the stimulus**: children overgeneralise rules ('goed', 'mouses') but never make wrong rule errors. Acquire structure-dependent rules without examples of errors.\n\n"
     "**Theories:**\n"
     "1. Nativism (Chomsky): innate Universal Grammar — language acquisition device constrains possible grammars\n"
     "2. Usage-based (Tomasello): no special faculty; general learning + pattern recognition + social attention\n"
     "3. Constructivist: some innate predispositions + general learning applied to input\n\n"
     "**Critical period**: Genie (no language exposure to age 13): never acquired normal grammar. Left hemisphere lateralisation stronger in early childhood.\n\n"
     "**Mechanisms**: statistical learning (8-month-olds track syllable transition probabilities), joint attention, motherese (simplified grammar + exaggerated prosody).",
     9.0, "Q446 Language acquisition: poverty of stimulus; Chomsky UG vs Tomasello usage-based; Genie critical period; statistical learning, joint attention, motherese"),

    (447, "Physics", "What is the photoelectric effect, and why did it matter for quantum mechanics?",
     "Light hits metal -> electrons ejected, BUT only above a threshold frequency. Below threshold: no electrons regardless of intensity.\n\n"
     "**Classical prediction**: intensity drives ejection. WRONG.\n"
     "**Observation**: energy of each electron depends only on FREQUENCY, not intensity. Intensity only affects number of electrons.\n\n"
     "**Einstein's explanation (1905, Nobel 1921)**: light comes as photons with energy E = hv. To eject electron, photon must have hv >= work function (Phi). KE = hv - Phi.\n\n"
     "**Why it mattered:**\n"
     "1. Confirmed Planck's energy quantisation (1900)\n"
     "2. Wave-particle duality: light is both\n"
     "3. Led to quantum mechanics: Bohr (1913), Heisenberg (1925), Schrodinger (1926)\n"
     "4. Technology: photovoltaics, CCDs, photomultipliers",
     9.5, "Q447 Photoelectric effect (Einstein 1905): frequency not intensity determines electron energy; E=hv; established wave-particle duality; foundation of quantum mechanics; Nobel 1921"),

    (448, "Sociology", "What is social mobility and what actually predicts it?",
     "Movement between socioeconomic strata. Intergenerational (parent to child) vs intragenerational (within lifetime).\n\n"
     "**Great Gatsby Curve** (Miles Corak): more income inequality -> LOWER intergenerational mobility. Denmark (low inequality) = high mobility; US, UK = lower.\n\n"
     "**Predictors (Chetty et al.):**\n"
     "1. Parental income/wealth: strongest predictor; ZIP code of birth major determinant of lifetime earnings\n"
     "2. School quality: access to good schools drives mobility\n"
     "3. Social capital/networks: who you know, unevenly distributed\n"
     "4. Geography: some US counties (Midwest, Salt Lake City) high mobility; parts of Southeast very low\n"
     "5. Neighbourhood at age <13: moving to better neighbourhood before 13 significantly improves outcomes (Opportunity Insights)\n\n"
     "Absolute mobility (exceeding parents in absolute terms) has historically been high; relative mobility much more constrained.",
     9.0, "Q448 Social mobility: Great Gatsby Curve (inequality->lower mobility); Chetty: ZIP code/neighbourhood at <13 key; school quality, social networks; Great Gatsby Curve"),

    (449, "Chemistry", "How do batteries work chemically, and what limits their energy density?",
     "Battery = chemical -> electrical energy via spatially separated redox reactions.\n"
     "Anode: oxidation (loses electrons). Cathode: reduction (gains electrons). Electrons flow externally (current); ions flow internally through electrolyte.\n\n"
     "**Li-ion battery:**\n"
     "- Anode: graphite (Li intercalates during charge)\n"
     "- Cathode: LiCoO2/LiFePO4/NMC (Li de-intercalates during charge)\n"
     "- Electrolyte: Li salt in organic solvent\n"
     "- Discharge: Li+ moves spontaneously to cathode; electrons flow through circuit\n\n"
     "**What limits energy density:**\n"
     "- Anode capacity: graphite 372 mAh/g; silicon 3,500 mAh/g but 300% volume expansion -> cracking\n"
     "- Cathode voltage limited by electrolyte/electrode stability\n"
     "- Dead weight: packaging, separator, electrolyte\n"
     "- Safety: thermal runaway risk\n\n"
     "**Theoretical limits**: Li-ion ~265 Wh/kg practical; Li-sulfur 2,600 Wh/kg; Li-air 11,680 Wh/kg (comparable to gasoline).",
     9.5, "Q449 Li-ion battery: anode (graphite intercalation) / cathode (LiCoO2) / Li salt electrolyte; limited by anode capacity (Si expands), dead weight; Li-air 11,680 Wh/kg theoretical"),

    (450, "General Knowledge", "What is the most important invention in human history, and why?",
     "Depends on dimension valued. Four strong candidates:\n\n"
     "**Language**: meta-invention enabling all others to accumulate. But may have evolved rather than been invented.\n\n"
     "**Writing (~3100 BCE, Mesopotamia)**: first technology for externalising memory. Enables laws, science, literature, accumulating civilisation. Without it, each generation relearns from scratch.\n\n"
     "**Printing press (~1450)**: multiplied writing's impact. Ideas at scale. Enabled Reformation, Scientific Revolution, democracy. 500-year acceleration of knowledge.\n\n"
     "**Scientific method (~17th century)**: not physical but a process — hypothesis, controlled experiment, replication, peer review. Converts curiosity into reliable knowledge.\n\n"
     "**Why printing press may win**: key multiplier. One idea -> millions of lives in years, not centuries. Created the feedback loop: print -> literacy -> scientific culture -> industrial revolution.\n\n"
     "**Honourable mentions**: fire, agriculture, germ theory, transistors.\n\n"
     "**Meta-point**: no invention stands alone. Cumulative innovation is humanity's real superpower.",
     9.0, "Q450 Most important invention: language (meta-invention), writing (3100 BCE, externalising memory), printing press (key multiplier, 1450), scientific method; printing press may win"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
