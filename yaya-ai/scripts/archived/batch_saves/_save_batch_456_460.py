"""Save Q456-Q460 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (456, "Chemistry", "What is entropy, and why does it always increase?",
     "Entropy (S) measures the number of microscopic arrangements (microstates) compatible with a macroscopic state. More arrangements = higher entropy.\n\n"
     "**Second Law of Thermodynamics**: in an isolated system, entropy never decreases. It increases or stays the same.\n\n"
     "**Why it increases**: probability. A gas released into a vacuum spreads out because the disordered states vastly outnumber the ordered ones -- not because of any force, but because disorder is overwhelmingly more likely. Boltzmann: S = k ln(W), where W is the number of microstates.\n\n"
     "**Examples**:\n"
     "- Ice melting: liquid water has more microstates than crystal ice\n"
     "- Perfume spreading: molecules explore all available space\n"
     "- Stars forming: locally decreases entropy (order from gas) but radiates far more entropy as heat\n\n"
     "**Arrow of time**: entropy gives time its direction. The laws of physics are symmetric -- they work the same forward and backward -- but entropy breaks that symmetry. The past is defined as lower entropy.\n\n"
     "**Maxwell's Demon**: thought experiment of a demon sorting fast/slow molecules to decrease entropy. Resolved by Szilard (1929) and Landauer (1961): the demon must erase memory, which costs entropy -- so the second law holds.\n\n"
     "**Heat death**: eventual equilibrium state of universe -- maximum entropy, no temperature differences, no useful work possible.",
     9.0, "Q456 Entropy: S=k ln(W); second law = entropy never decreases in isolated system; probability not force; arrow of time; Maxwell's Demon resolved by Landauer"),

    (457, "Political Science", "What is democracy and why do some democracies fail?",
     "Democracy: a system where political power derives from the governed, expressed through free elections, rule of law, and protection of rights.\n\n"
     "**Types**:\n"
     "- Direct (Athens): citizens vote on every decision. Impractical at scale.\n"
     "- Representative: elect representatives who govern. Most modern democracies.\n"
     "- Liberal democracy: adds constraints on majority power -- minority rights, press freedom, independent judiciary.\n\n"
     "**Why democracies fail (Levitsky & Ziblatt, How Democracies Die, 2018)**:\n"
     "1. **Executive aggrandisement**: elected leaders gradually grab power -- filling courts, weakening press, harassing opponents. Legal but antidemocratic.\n"
     "2. **Polarisation**: when parties see each other as existential enemies, norms of mutual toleration break down -- 'anything goes to win.'\n"
     "3. **Weak institutions**: democracies require strong courts, free press, civil society. Without them, leaders face no check.\n"
     "4. **Economic crisis**: Weimar Germany, Venezuela -- economic collapse drives voters to authoritarians promising order.\n\n"
     "**Historical record**: Freedom House 2023 -- 17th consecutive year of democratic decline globally. Hungary and Turkey went from democracy to competitive authoritarianism under elected leaders.\n\n"
     "**What sustains it**: independent judiciary, civilian control of military, strong civic culture, two-party or multi-party competition, peaceful transfers of power as norm.",
     9.0, "Q457 Democracy: direct/representative/liberal; fails via executive aggrandisement, polarisation, weak institutions (Levitsky 2018); 17yr decline globally"),

    (458, "Nutrition", "What does the science actually say about diet and health?",
     "Nutrition science is hard: humans can't be randomised to diets for decades, self-reporting is unreliable, and food is chemically complex.\n\n"
     "**What the evidence actually supports**:\n"
     "1. **Mediterranean diet**: highest quality evidence (PREDIMED RCT, 2013) -- reduces cardiovascular events ~30%. Olive oil, vegetables, fish, nuts, moderate wine.\n"
     "2. **Ultra-processed foods (UPF)**: Nova classification. UPFs linked to obesity, T2 diabetes, depression, all-cause mortality in cohort studies. Mechanisms unclear -- could be additives, processing, or displacement of whole foods.\n"
     "3. **Fibre**: consistently beneficial -- gut microbiome, colorectal cancer risk, satiety, blood glucose control.\n"
     "4. **Added sugar**: strong evidence for metabolic harm at high doses (T2 diabetes, NAFLD).\n"
     "5. **Saturated fat**: more complicated than '70s advice -- butter vs processed meat differ. Total fat less important than fat type.\n\n"
     "**Contested**:\n"
     "- Red meat: processed clearly harmful; unprocessed uncertain\n"
     "- Eggs: probably fine for most people\n"
     "- Dairy: no clear harm in most evidence\n"
     "- Low-carb vs low-fat: similar outcomes at 1-2 years; adherence is the main driver\n\n"
     "**What we don't know**: optimal macronutrient ratio, individual variation (gut microbiome, genetics), whether associations are causal.\n\n"
     "**Consensus**: varied whole foods, mostly plants, minimal processing, limited added sugar. Almost no controversy about that.",
     8.5, "Q458 Diet science: Mediterranean diet (PREDIMED RCT), fibre, UPF harm, added sugar; low-carb vs low-fat similar; adherence matters most; consensus = whole foods"),

    (459, "Mathematics", "What is Godel's incompleteness theorem and why does it matter?",
     "Godel (1931): any consistent formal system powerful enough to describe basic arithmetic contains true statements that cannot be proved within that system.\n\n"
     "**Two theorems**:\n"
     "1. **First incompleteness**: for any consistent, sufficiently powerful formal system F, there exists a statement G ('I am not provable in F') that is true but unprovable in F.\n"
     "2. **Second incompleteness**: F cannot prove its own consistency.\n\n"
     "**How he did it**: Godel numbering -- encode every formula and proof as a number. Then construct a self-referential formula G that says 'the formula with Godel number X is not provable' where X is the Godel number of G itself. If F is consistent, G can't be proved; if F proves G, F is inconsistent.\n\n"
     "**Why it matters**:\n"
     "- Killed Hilbert's programme: math cannot be fully formalised into one complete, consistent system\n"
     "- Not 'math is broken' -- it means no single formal system captures all mathematical truth\n"
     "- Human mathematicians vs computers: some argue Godel implies human intuition transcends formal systems (Penrose). Contested.\n\n"
     "**Common misconceptions**:\n"
     "- Doesn't mean 'everything is unprovable' -- most math is provable\n"
     "- Doesn't mean truth is relative\n"
     "- Doesn't apply to every formal system -- only those that include arithmetic\n\n"
     "**Impact**: foundations of mathematics, computer science (Turing's halting problem is equivalent), philosophy of mind.",
     10.0, "Q459 Godel incompleteness (1931): consistent arithmetic systems have true unprovable statements; killed Hilbert programme; Godel numbering; not 'math is broken'"),

    (460, "Anthropology", "What is culture and how does it shape human behaviour?",
     "Culture: the shared beliefs, practices, symbols, values, and norms transmitted across generations within a group. Humans are uniquely cultural animals -- we inherit not just genes but accumulated knowledge.\n\n"
     "**Cultural universals** (Brown 1991): every human culture has language, music, kinship systems, rituals, tools, religion/supernatural beliefs, taboos. But the specific forms vary enormously.\n\n"
     "**How culture shapes behaviour**:\n"
     "1. **Cognition**: WEIRD (Western, Educated, Industrial, Rich, Democratic) samples dominate psychology research. Cross-cultural tests show different: perception (Muller-Lyer illusion less effective in non-WEIRD populations), reasoning (analytic vs holistic), self-concept (independent vs interdependent).\n"
     "2. **Emotion**: Ekman's 6 basic emotions (fear, anger, joy, sadness, disgust, surprise) show universal facial expressions, but emotional display rules and experience vary. Inuit have many words for snow-related emotional states; some cultures lack a concept of 'depression.'\n"
     "3. **Morality**: Haidt's moral foundations (care, fairness, loyalty, authority, purity) -- cultures weight them differently. Individualist cultures emphasise care/fairness; collectivist add loyalty/authority.\n\n"
     "**Nature vs culture**: not a binary. Gene-culture coevolution -- lactase persistence spread where cattle herding developed. Culture shapes which genes are selected.\n\n"
     "**Cultural change**: cultures aren't static. Contact, trade, media, migration all accelerate change. Globalisation homogenises some aspects (English, jeans, smartphones) while local cultures resist and adapt.",
     9.0, "Q460 Culture: shared beliefs/practices across generations; universals (Brown 1991) with variable forms; WEIRD bias in psychology; gene-culture coevolution; not static"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
