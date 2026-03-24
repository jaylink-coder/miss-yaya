"""Save Q431-Q435 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (431, "Astronomy", "What are neutron stars and how do they form?",
     "Massive star (8-20 solar masses) core collapses. If core mass 1.4-3 solar masses, electron degeneracy fails. Protons + electrons merge (inverse beta decay) -> neutron star.\n\n"
     "**Properties:**\n"
     "- Diameter ~20km; mass ~1.4-2 solar masses; density ~10^17 kg/m^3 (teaspoon ~10M tonnes)\n"
     "- Escape velocity ~0.6c; magnetic fields up to 10^15 gauss (magnetars)\n"
     "- Spin: conservation of angular momentum; radius shrinks 100x -> spin increases 10,000x -> hundreds of rotations/second\n\n"
     "**Types:**\n"
     "- Pulsars: radio beam lighthouses; first discovered by Bell Burnell (1967); millisecond pulsars rival atomic clocks\n"
     "- Magnetars: extremely strong magnetic fields; starquake gamma-ray flares\n\n"
     "**Neutron star mergers (GW170817, 2017)**: first gravitational wave + electromagnetic detection. Kilonova: r-process nucleosynthesis produces gold, platinum, uranium. Primary source of heavy elements.",
     9.5, "Q431 Neutron stars: inverse beta decay, ~20km diameter, ~10^17 kg/m^3; pulsars (Bell Burnell 1967); GW170817 merger = kilonova = r-process heavy elements (gold)"),

    (432, "Psychology", "What is the Dunning-Kruger effect, and is it as universal as people think?",
     "Kruger & Dunning (1999): people with limited knowledge overestimate competence; experts underestimate relative ability (task seems easy to them too).\n\n"
     "**Original study**: bottom quartile performers on logic/grammar/humour overestimated by ~50 percentile points.\n\n"
     "**Replication and critique:**\n"
     "- Basic finding replicates\n"
     "- Statistical artifact critique (Gignac & Zajenkowski 2020): DK pattern can emerge from mathematical regression to the mean in ANY self-report + performance dataset, even if people perfectly track ability. Floor/ceiling effects.\n"
     "- Effect is real but smaller than popular 'stupid people are confidently wrong' narrative\n\n"
     "**What it captures:**\n"
     "- Metacognition is hard (knowing what you don't know requires knowing what you don't know)\n"
     "- Novices use wrong framework to evaluate own performance\n"
     "- Domain-specific: expert engineer may still have DK in social skills",
     8.5, "Q432 Dunning-Kruger (1999): low-skill overestimate, expert underestimate; statistical artifact critique (regression to mean); effect real but smaller than popular narrative; domain-specific"),

    (433, "History", "Why did the Roman Empire fall?",
     "Western Empire fell 476 CE (Romulus Augustulus deposed). Eastern Empire (Byzantium) lasted to 1453. No single cause:\n\n"
     "**Military/political:** overextended borders; Germanic foederati with divided loyalty; political instability (50 emperors in 50 years, 235-284 CE); military power to make/unmake emperors.\n\n"
     "**Economic:** currency debasement -> inflation; tax burden drove avoidance; declining long-distance trade.\n\n"
     "**External pressures:** Huns from east displaced Germanic tribes westward (Goths, Vandals, Franks); Sassanid Persia contested east.\n\n"
     "**Historiographical positions:**\n"
     "- Heather: military pressure primary; Rome healthy until external shocks overwhelmed it\n"
     "- Ward-Perkins: real material decline (pottery, literacy, building quality collapse)\n"
     "- Brown: 'transformation' not collapse — Roman culture merged with Germanic and Christian\n\n"
     "**What didn't fall**: Eastern Empire thrived. Fall was primarily western.",
     9.0, "Q433 Roman Empire fall (476 CE): military overextension, political instability, currency debasement, barbarian pressure (Huns displaced Goths); Heather vs Ward-Perkins vs Brown"),

    (434, "Technology", "How does GPS actually work?",
     "24-32 satellites at ~20,200km; 6 orbital planes; each completes 2 orbits/day.\n\n"
     "**Core principle: trilateration via time signals.**\n"
     "Satellite broadcasts position + precise time. Receiver measures signal travel time. Distance = c * time. 3 satellites -> 3 spheres -> intersection. Cheap receiver clock -> 4th satellite to solve 4 unknowns (x, y, z, time error).\n\n"
     "**Accuracy:**\n"
     "- Standard GPS: 3-5m\n"
     "- Differential GPS (ground stations): <1m\n"
     "- RTK (real-time kinematic, surveying): centimetre-level\n"
     "- Ionospheric delay corrected by dual frequency (L1/L2)\n\n"
     "**Relativity corrections (built in):**\n"
     "- Weaker gravity: satellites' clocks run faster +45 microseconds/day (GR)\n"
     "- High velocity: clocks run slower -7 microseconds/day (SR)\n"
     "- Net: +38 microseconds/day. Without correction: GPS drifts ~10km/day. Einstein's corrections are in the system.",
     9.5, "Q434 GPS: trilateration from 4 satellites (3 position + 1 time correction); 3-5m standard accuracy; relativity corrections required (+38 microseconds/day without = 10km/day drift)"),

    (435, "Biology", "What is epigenetics and does it challenge our understanding of inheritance?",
     "Heritable changes in gene expression without DNA sequence changes. Genome = instruction manual; epigenome = which instructions are read.\n\n"
     "**Mechanisms:**\n"
     "1. DNA methylation: CH3 groups on cytosines at CpG sites; methylated promoters silence genes\n"
     "2. Histone modification: DNA wrapped around histones; acetylation opens chromatin (more expression); methylation activates or represses; the 'histone code'\n"
     "3. Non-coding RNA (miRNA, lncRNA): regulate mRNA stability and translation\n\n"
     "**Why it matters:** differentiation (same DNA, different cells), cancer (tumour suppressors silenced), environment response (stress, diet, smoking alter marks).\n\n"
     "**Transgenerational inheritance:** Dutch Hunger Winter (1944-45) — children of starved mothers had higher obesity/diabetes decades later. But methylation largely RESET at fertilisation (epigenetic reprogramming). Transgenerational inheritance in mammals limited and debated.\n\n"
     "Does NOT validate Lamarck — mechanisms are different from direct inheritance of acquired characteristics.",
     9.5, "Q435 Epigenetics: DNA methylation + histone code + ncRNA; same DNA, different cells; Dutch Hunger Winter transgenerational effects; reset at fertilisation; not Lamarckism"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
