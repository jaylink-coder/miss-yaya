"""Save Q406-Q410 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (406, "Chemistry", "What is entropy and why does it always increase?",
     "Entropy (S) measures the number of microscopic arrangements (microstates) producing the same macroscopic state. Boltzmann: S = k*ln(W).\n\n"
     "**Why it increases**: pure statistics. Vastly more disordered arrangements exist than ordered ones. Gas expanding into vacuum isn't forced — it's just that 10^23 molecules spontaneously re-compressing is so improbable it never happens on cosmological timescales.\n\n"
     "**Second Law**: in an isolated system, entropy never decreases. Total entropy of the universe increases in any irreversible process.\n\n"
     "**Clarifications:**\n"
     "- Local entropy CAN decrease (refrigerator, crystal, life) — but only by increasing entropy elsewhere by more\n"
     "- Earth takes in low-entropy sunlight, radiates high-entropy heat\n"
     "- Time's arrow: Second Law is the only fundamental physics law distinguishing past from future\n\n"
     "**Maxwell's demon** (Landauer 1961): demon's memory erasure costs at least k*T*ln(2) per bit — net entropy still increases. Information and thermodynamics deeply linked.",
     9.5, "Q406 Entropy: S=k*ln(W), increases by statistics not physical law; time's arrow; Maxwell's demon resolved by Landauer (information = thermodynamics)"),

    (407, "Psychology", "What is cognitive dissonance and how do people actually resolve it?",
     "Festinger (1957): holding two conflicting cognitions creates psychological discomfort. Three resolution strategies:\n\n"
     "1. **Change a belief**: smoker convinces themselves they're immune.\n"
     "2. **Change behaviour**: actually quit. Hardest.\n"
     "3. **Add cognitions**: 'smoking reduces stress, which also causes cancer — net neutral.'\n\n"
     "**Classic experiment** (Festinger & Carlsmith 1959): paid $1 or $20 to lie a boring task was interesting. $1 group later genuinely rated task as interesting — they had to justify compliance, so changed attitude. *Behaviour precedes belief when external justification is low.*\n\n"
     "**Post-purchase rationalisation**: after buying, people seek confirming info and dismiss negatives.\n\n"
     "**Belief perseverance** (When Prophecy Fails — doomsday cult): when predicted disaster didn't occur, members became *more* fervent, not less. Dissonance resolved by increasing commitment.\n\n"
     "Dissonance magnitude: greater when decisions are important, irreversible, options similar in quality.",
     9.0, "Q407 Cognitive dissonance (Festinger 1957): resolve by changing belief/behaviour/adding cognitions; $1 experiment: low justification -> attitude change; doomsday cult -> more fervent"),

    (408, "Astronomy", "How do astronomers know the age of the universe is 13.8 billion years?",
     "Multiple independent methods converge:\n\n"
     "1. **CMB (Planck)**: temperature fluctuation power spectrum fit to LCDM model gives 13.787 +/- 0.020 Gyr\n"
     "2. **Hubble constant + expansion history**: H0 combined with Friedmann equations gives look-back time. Problem: H0 tension (local ~73 vs CMB ~67.4 km/s/Mpc) — unresolved\n"
     "3. **Oldest stars**: globular clusters; HD 140283 'Methuselah star' = 14.46 +/- 0.8 Gyr — consistent within error\n"
     "4. **Radioactive dating (cosmochronometry)**: U/Th ratios in old stars give ~12-14 Gyr, independent of expansion models\n\n"
     "**Why trust it**: four independent methods (CMB, expansion history, stellar ages, nuclear physics) all point to 13-14 Gyr. Concordance is extraordinary.",
     9.5, "Q408 Universe age 13.8 Gyr: CMB (Planck), Hubble expansion, oldest stars, radioactive dating — all independent, all agree; H0 tension unresolved"),

    (409, "Law", "What is the difference between civil law and common law legal systems?",
     "**Common law** (UK, US, Australia, Canada, India):\n"
     "- Judge-made law; past decisions (stare decisis) bind future courts\n"
     "- Law develops case-by-case\n"
     "- Adversarial: two parties argue before neutral judge\n"
     "- Juries common in criminal and civil cases\n\n"
     "**Civil law** (France, Germany, Europe, Latin America, Japan):\n"
     "- Comprehensive codes from Roman law (Justinian's Corpus Juris Civilis)\n"
     "- Statutes primary; judges interpret, don't create\n"
     "- Inquisitorial: judge actively investigates\n"
     "- Less precedent; fewer juries\n\n"
     "**Key practical differences:**\n"
     "- Discovery: US allows extensive pre-trial evidence exchange; civil law more limited\n"
     "- Contract interpretation: common law literal text; civil law good faith and purpose\n"
     "- Torts: common law case-by-case; civil law general clauses (French Art. 1382)\n\n"
     "**Hybrid**: Louisiana, Quebec, Scotland — civil law base with common law influence.",
     8.5, "Q409 Common law (precedent, adversarial, UK/US) vs civil law (codes, inquisitorial, France/Europe); hybrids: Louisiana, Quebec, Scotland"),

    (410, "Technology", "What is Moore's Law and is it still true?",
     "Gordon Moore (1965): transistors on a chip double every ~2 years at same cost. Not physical law — empirical observation, self-fulfilling prophecy shaping semiconductor roadmaps.\n\n"
     "**Historical accuracy**: held ~1965-2015. From 4 transistors (1959) to 10B+ (modern).\n\n"
     "**What changed:**\n"
     "- Physical limits: transistors at 3nm (~10 atoms). Quantum tunnelling, heat, manufacturing variability\n"
     "- Dennard scaling ended ~2006: smaller no longer automatically faster or cooler => multi-core\n"
     "- Economic: new fabs cost $20B+\n"
     "- Slowing: doubling now takes 3+ years\n\n"
     "**What continues:**\n"
     "- 3D stacking: NAND, HBM, 3D-IC — vertical scaling\n"
     "- Specialised silicon: GPUs, TPUs, NPUs bypass general-purpose scaling\n"
     "- Chiplets: multiple smaller dies in one package\n\n"
     "**Verdict**: strict transistor-count doubling has slowed. 'Moore's Law is dead' is too strong — relentless compute improvement continues through different means.",
     9.0, "Q410 Moore's Law (1965): transistors double ~2yr; slowing since ~2015; Dennard scaling ended; continues via 3D stacking, specialised silicon (GPUs/TPUs), chiplets"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
