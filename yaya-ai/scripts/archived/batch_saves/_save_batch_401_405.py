"""Save Q401-Q405 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (401, "Physics", "What is quantum entanglement and does it allow faster-than-light communication?",
     "Two particles share a quantum state such that measuring one instantly determines the other, regardless of distance.\n\n"
     "**Mechanism**: two photons in superposition |up-down> + |down-up>. Neither is definite. Measure one as spin-up, the other instantly becomes spin-down. Einstein called this 'spooky action at a distance.'\n\n"
     "**Bell's theorem** (1964): local hidden variable theories can't explain quantum correlations. Bell inequality violations confirmed experimentally (Aspect 1982; loophole-free Hensen 2015) — correlations are genuinely non-local.\n\n"
     "**But no FTL communication** (no-communication theorem): you can't control which outcome you get — results are random. Correlations only appear when you compare results via classical communication (limited to c).\n\n"
     "**Applications**: quantum computing (entanglement enables speedup), quantum cryptography (BB84 — eavesdropping disturbs entanglement, detectable), quantum teleportation (transfers quantum state, not matter, still needs classical channel).",
     9.5, "Q401 Quantum entanglement: real non-local correlations (Bell violations confirmed); no FTL communication (no-communication theorem); used in QC, cryptography"),

    (402, "Economics", "What is comparative advantage and why does it justify trade even when one country is better at everything?",
     "Ricardo (1817): even if one country is absolutely better at everything, both benefit from specialising based on *relative* efficiency.\n\n"
     "**Example**: England (100h cloth, 120h wine) vs Portugal (90h cloth, 80h wine). Portugal better at both. But opportunity cost: Portugal gives up 0.89 cloth per wine; England gives up 1.2. Portugal has comparative advantage in wine, England in cloth.\n\n"
     "**Mechanism**: opportunity cost, not absolute productivity, determines gains from trade. Both countries can consume more of both goods by specialising.\n\n"
     "**Complications**: assumes full employment (fails short-term); ignores adjustment costs; dynamic effects (infant industry argument); doesn't distribute gains equally — trade creates winners and losers within countries.\n\n"
     "**Still valid**: explains why nearly all economists support free trade. Political problem is distributional, not efficiency-based.",
     9.0, "Q402 Comparative advantage (Ricardo 1817): relative not absolute efficiency; opportunity cost determines specialisation; trade creates winners and losers"),

    (403, "Neuroscience", "What happens in the brain during sleep, and why do we need it?",
     "Sleep is not passive. The brain cycles through NREM and REM in ~90-minute cycles.\n\n"
     "**Stages**: NREM1 (drowsy), NREM2 (sleep spindles, K-complexes — motor memory), NREM3/slow-wave (delta waves, growth hormone, declarative memory consolidation), REM (brain active as waking, body paralysed — dreams, emotional memory).\n\n"
     "**Why we need it:**\n"
     "1. Memory consolidation: hippocampus replays experiences, transfers to cortex during slow-wave (Stickgold 2005)\n"
     "2. Glymphatic clearance: CSF flushes beta-amyloid and tau during sleep (Xie 2013) — disrupted in Alzheimer's\n"
     "3. Synaptic homeostasis (Tononi): sleep downscales synapses ~20%, maintaining signal-to-noise ratio\n"
     "4. Immune function: cytokine production, T-cell activation\n"
     "5. Metabolic restoration: glucose regulation, ATP\n\n"
     "**Sleep deprivation**: 17-19h awake impairs like 0.05% BAC. 24h like 0.10%. Fatal familial insomnia: prion destroys sleep circuits, death in months.",
     9.5, "Q403 Sleep: NREM/REM cycles; memory consolidation, glymphatic clearance (beta-amyloid), synaptic homeostasis, immune function; deprivation as bad as alcohol"),

    (404, "Mathematics", "What is the Riemann Hypothesis and why do mathematicians care so much about it?",
     "Riemann zeta function: z(s) = 1 + 1/2^s + 1/3^s + ... extended to all complex numbers.\n\n"
     "Riemann (1859): trivial zeros at s = -2, -4, -6...; infinitely many non-trivial zeros in critical strip 0 < Re(s) < 1.\n\n"
     "**Hypothesis**: all non-trivial zeros lie on Re(s) = 1/2.\n\n"
     "**Why it matters:**\n"
     "- Prime number distribution: zeros control error in Prime Number Theorem. RH true => tightest possible bound O(sqrt(x) log x)\n"
     "- 1000+ theorems conditional on RH\n"
     "- Cryptography: prime distribution affects RSA security\n"
     "- Physics: zero spacing matches eigenvalues of random Hermitian matrices (GUE) — mysterious connection to quantum chaos\n\n"
     "**Status**: verified for first 10^13 zeros; no counterexample; no proof. Clay Millennium Problem ($1M prize). One of hardest open problems in mathematics.",
     10.0, "Q404 Riemann Hypothesis: non-trivial zeros of zeta on Re(s)=1/2; controls prime distribution; 1000+ conditional theorems; Millennium Problem"),

    (405, "Sociology", "What is social capital, and does it actually predict outcomes in the real world?",
     "Social capital (Bourdieu, Coleman, Putnam): value embedded in social networks — trust, reciprocity norms, resource access through relationships.\n\n"
     "**Three types (Putnam):**\n"
     "- Bonding: within tight groups (family). Good for hard times; can exclude outsiders.\n"
     "- Bridging: across groups. More economically valuable — weak ties (Granovetter 1973) are how most people find jobs.\n"
     "- Linking: ties to institutions and authority.\n\n"
     "**Evidence:**\n"
     "- Health: Berkman & Syme 1979 — social isolation as deadly as 15 cigarettes/day.\n"
     "- Crime: Sampson — neighbourhood collective efficacy predicts crime even controlling for poverty.\n"
     "- Economic growth: Knack & Keefer 1997 — trust in strangers correlates with GDP growth.\n"
     "- Italy: Putnam — regional governance quality mapped to medieval civic associational life.\n\n"
     "**Critique**: causation hard to establish; reinforces inequality (who has high-value networks?); Bourdieu frames it as class reproduction mechanism.",
     9.0, "Q405 Social capital (Putnam): bonding/bridging/linking; predicts health, crime, GDP, education; weak ties key (Granovetter); Bourdieu = class reproduction"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
