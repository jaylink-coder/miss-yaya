"""Save Q441-Q445 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (441, "Medicine", "What is antibiotic resistance and how does it develop?",
     "Antibiotics target cell wall (penicillin), protein synthesis (tetracycline), DNA replication (fluoroquinolones), or membranes. Resistance = survival.\n\n"
     "**Darwinian mechanism:**\n"
     "1. Rare resistant mutants in population\n"
     "2. Antibiotic kills susceptible bacteria, resistant survive\n"
     "3. Resistant reproduce (bacteria: 20 min/generation, ~500 generations/week)\n"
     "4. Horizontal gene transfer: resistance genes spread on plasmids across species (MRSA: mecA gene)\n\n"
     "**Mechanisms:** efflux pumps (expel antibiotic), enzyme degradation (beta-lactamases cleave penicillin ring), target modification, reduced permeability.\n\n"
     "**Scale**: AMR kills ~1.27M/year directly; WHO projects 10M/year by 2050 without action.\n\n"
     "**Drivers**: unnecessary prescriptions, livestock use, incomplete courses, broad-spectrum overuse.\n\n"
     "**Why few new antibiotics**: 10-15yr development; brief low-cost use = poor financial incentive vs lifelong drugs.",
     9.5, "Q441 Antibiotic resistance: Darwinian selection + horizontal gene transfer (plasmids); efflux pumps, beta-lactamases, target modification; 1.27M deaths/year; 10M/year by 2050"),

    (442, "History", "What was the significance of the Magna Carta?",
     "June 1215: English barons forced King John to sign. Originally limited royal power in narrow feudal ways — mostly barons' rights, not ordinary people.\n\n"
     "**Key clauses**: Cl. 39 — no free man imprisoned without lawful judgment of peers ('due process'); Cl. 40 — no selling/denying justice.\n\n"
     "**At the time**: John got Pope Innocent III to annul it immediately. Civil war resumed. Most clauses dropped by 1217.\n\n"
     "**Why it mattered more later**: 17th century — Coke and Selden used it against Stuart prerogatives. It became a *symbol*: king is under the law. American founders invoked it; 5th Amendment due process draws on Cl. 39.\n\n"
     "**What it didn't do**: 'free man' in 1215 = barons and knights, not 90% who were serfs. Didn't protect women.\n\n"
     "**Lasting principle**: rule of law over arbitrary power -> constitutional government, habeas corpus, human rights.",
     9.0, "Q442 Magna Carta (1215): barons limited King John; Cl. 39 due process; immediately annulled by Pope; later became symbol of rule of law; influenced 5th Amendment, habeas corpus"),

    (443, "Computer Science", "What is machine learning and how is it different from traditional programming?",
     "**Traditional programming**: programmer writes explicit rules. Input + rules -> output. Brittle; can't generalise.\n\n"
     "**Machine learning**: provide data + correct answers; algorithm finds the rules. Input + labels -> model -> generalises to new data.\n\n"
     "**Core paradigms:**\n"
     "1. Supervised learning: labelled data (x, y) -> learn f(x)=y; classification, regression\n"
     "2. Unsupervised: unlabelled data; find structure; clustering, dimensionality reduction (PCA, t-SNE)\n"
     "3. Reinforcement learning: agent + rewards -> policy; AlphaGo, robotics\n\n"
     "**Why it works**: neural networks approximate arbitrarily complex functions (universal approximation theorem); deep learning learns hierarchical features automatically.\n\n"
     "**What it can't do (yet)**: causal reasoning, robust out-of-distribution generalisation, reliable explainability.\n\n"
     "**Scale**: LLMs (GPT, Claude) = transformers on trillions of tokens; emergent capabilities at scale.",
     9.0, "Q442 Machine learning vs traditional programming: algorithm finds rules from data; supervised/unsupervised/RL; universal approximation; LLMs = transformers at scale; can't reason causally"),

    (444, "Philosophy", "What is the ship of Theseus paradox, and what is it really asking?",
     "Plutarch (~100 CE): every plank of Theseus's ship replaced over time. Still the same ship?\n\n"
     "**Hobbes extension**: original planks collected and reassembled. Two ships claim to be the original. Which is it?\n\n"
     "**What it's really asking**: principle of identity over time — what makes X at T1 the 'same' as Y at T2?\n\n"
     "**Theories:**\n"
     "1. Spatiotemporal continuity: repaired ship has unbroken causal chain — same ship\n"
     "2. Compositional identity: original planks = original ship\n"
     "3. Functional/social: the one people use and call it is the ship\n"
     "4. Four-dimensionalism: objects extended through time; 'same' selects different time-slices\n\n"
     "**Real applications**: personal identity (we replace all atoms every ~7 years), corporate identity, legal identity of nations.\n\n"
     "**Parfit** (*Reasons and Persons*): personal identity is not what matters — psychological continuity is what matters; 'same' may be an empty question.",
     9.5, "Q444 Ship of Theseus (Plutarch): identity over time; spatiotemporal continuity vs composition vs function vs four-dimensionalism; Parfit: identity may be empty, psychological continuity matters"),

    (445, "Music", "Why does music make us emotional?",
     "Multiple converging mechanisms:\n\n"
     "1. **Expectation/violation** (Meyer 1956, Huron 2006): music is temporal prediction. Correct resolution = dopaminergic reward. Surprise + resolution = stronger reward. Chills = surprising resolution.\n\n"
     "2. **Neurobiology**: music activates nucleus accumbens + VTA (same as food/sex). Salimpoor 2011 fMRI: dopamine at peak emotional moments. Naltrexone reduces musical pleasure.\n\n"
     "3. **Emotional contagion**: music mimics prosody of emotional speech. Fast/high = happy; slow/descending = sad. We 'catch' the emotion.\n\n"
     "4. **Autobiographical memory**: music powerfully linked to memories via conditioning. Retrieves not just memory but its emotion.\n\n"
     "5. **Physical entrainment**: rhythm entrains motor cortex + cerebellum. Involuntary foot-tapping. Bodily coupling has emotional valence.\n\n"
     "6. **Universal tendencies**: minor keys, slow tempo, dissonance produce sad/tense responses cross-culturally.\n\n"
     "**Mystery of sad music pleasure**: safe simulation, hormonal empathy reward, artistic resolution — distinct emotion from actual grief.",
     9.5, "Q445 Music emotion: expectation/violation (dopamine at resolution); nucleus accumbens + opioids; emotional contagion (mimics speech prosody); autobiographical memory; motor entrainment"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
