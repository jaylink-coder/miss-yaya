"""Save Q496-Q500 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (496, "Neuroscience", "How does sleep work and why do we need it?",
     "Sleep: a reversible state of reduced consciousness and metabolic activity occupying roughly a third of human life. Not passive rest -- an active, essential biological process.\n\n"
     "**Architecture**: sleep cycles ~90 minutes, repeating 4-6 times per night.\n"
     "- NREM Stage 1 (N1): light sleep, hypnic jerks, theta waves\n"
     "- NREM Stage 2 (N2): sleep spindles, K-complexes; bulk of sleep\n"
     "- NREM Stage 3 (N3): slow-wave sleep (SWS); delta waves; deepest; hardest to wake from\n"
     "- REM: rapid eye movement; dreaming; body paralysed (atonia); brain active as waking\n"
     "Earlier cycles: more SWS. Later cycles: more REM.\n\n"
     "**What sleep does**:\n"
     "1. **Memory consolidation**: hippocampus replays waking experiences during SWS; transfers to cortex. REM strengthens emotional memories and creative associations.\n"
     "2. **Glymphatic clearance**: during SWS, cerebrospinal fluid flushes through brain, clearing metabolic waste including amyloid-beta (Alzheimer's protein). Brain shrinks ~20% during sleep to allow flow.\n"
     "3. **Immune function**: cytokine production peaks during sleep; sleep deprivation impairs vaccine response by 50%.\n"
     "4. **Metabolic regulation**: leptin/ghrelin balance disrupted by poor sleep -- increases hunger, reduces satiety.\n"
     "5. **Cellular repair**: growth hormone released during SWS; protein synthesis, tissue repair.\n\n"
     "**Sleep deprivation effects**: 24h awake = legally drunk (0.10% BAC equivalent performance). 17 days of restricted sleep: cognitive performance equivalent to 48h total deprivation. Chronic restriction not fully recoverable with one night.\n\n"
     "**Why we dream (REM)**: uncertain. Theories: memory consolidation; threat simulation (Revonsuo); emotional processing (Walker); random activation (Hobson's activation-synthesis). No consensus.\n\n"
     "**Circadian rhythm**: SCN (suprachiasmatic nucleus) in hypothalamus is master clock; light resets it via retinal ganglion cells (ipRGCs). Melatonin signals darkness. Blue light at night delays melatonin by up to 3 hours.",
     9.5, "Q496 Sleep: NREM/REM architecture; memory consolidation, glymphatic clearance, immune function; 24h awake = drunk; circadian rhythm via SCN and melatonin; REM theories unclear"),

    (497, "Philosophy", "What is the philosophy of science — what makes something scientific?",
     "Philosophy of science: the study of the methods, foundations, and implications of science. Its central question: what distinguishes science from non-science?\n\n"
     "**Logical positivism** (Vienna Circle, 1920s): only statements verifiable by experience are meaningful. 'God exists' is meaningless; '2+2=4' is analytic; 'water boils at 100C' is verifiable. Problem: the verification principle itself is not verifiable.\n\n"
     "**Falsificationism** (Popper 1934): science cannot verify theories -- only falsify them. A theory is scientific iff it makes predictions that could in principle be proven wrong. 'All swans are white' is falsifiable; 'God works in mysterious ways' is not. Good science actively tries to disprove itself.\n"
     "Problem: the Duhem-Quine thesis -- any observation can be 'saved' by adjusting auxiliary hypotheses. No single test definitively refutes a theory.\n\n"
     "**Paradigm shifts** (Kuhn 1962): science proceeds through normal science (puzzle-solving within a paradigm) punctuated by revolutions (paradigm shifts). Scientists resist anomalies; eventually accumulate until a new framework wins. Implication: science is a social activity, not purely logical.\n\n"
     "**Lakatos**: research programmes have a hard core (unfalsifiable assumptions) protected by a belt of auxiliary hypotheses. Progressive programmes make novel predictions; degenerative ones just patch anomalies.\n\n"
     "**Feyerabend**: 'anything goes' -- no universal scientific method. History of science shows successful violations of every proposed rule.\n\n"
     "**Demarcation problem**: what marks the line? Astrology makes predictions but fails them. Psychoanalysis is unfalsifiable. Creation science mimics scientific form. No single criterion suffices.\n\n"
     "**Modern consensus**: science is characterised by: testable predictions, peer review, replication, willingness to revise, cumulative progress. It is a method, not a body of facts.",
     9.0, "Q497 Philosophy of science: positivism -> Popper falsification -> Kuhn paradigms -> Lakatos research programmes; demarcation problem; Feyerabend 'anything goes'; science = method"),

    (498, "History", "What was the Industrial Revolution and how did it change human life?",
     "The Industrial Revolution (Britain ~1760-1840, spreading globally to 1900): the transition from agrarian, handicraft economies to manufacturing, machine production, and urban society. The most fundamental transformation of human material life since agriculture.\n\n"
     "**Key technologies**:\n"
     "- Steam engine (Watt 1769, improved 1782): applied to textile mills, mines, railways, ships\n"
     "- Spinning jenny (Hargreaves 1764), water frame (Arkwright 1769): mechanised textile production\n"
     "- Puddling process (Cort 1784): mass production of wrought iron\n"
     "- Steam locomotive (Stephenson's Rocket 1829): railways transformed movement of goods and people\n\n"
     "**Why Britain first?**\n"
     "1. Coal and iron geographically co-located\n"
     "2. Secure property rights, patent system\n"
     "3. Empire provided raw materials and markets\n"
     "4. Agricultural revolution freed labour for factories\n"
     "5. Canal system already existed for bulk transport\n\n"
     "**How it changed human life**:\n"
     "- Urbanisation: Britain 20% urban in 1800, 70% by 1900. Manchester grew from 25,000 to 350,000 in 100 years.\n"
     "- Living standards: initially fell for factory workers (child labour, 14-hour days, dangerous conditions). Rose substantially by late 19th century -- real wages doubled 1820-1870.\n"
     "- Life expectancy: urban death rates initially ROSE (cholera, typhoid, overcrowding). Sanitation reforms from 1850s reversed this.\n"
     "- Family structure: nuclear family; women and children entered workforce; later withdrew to 'separate spheres.'\n"
     "- Time discipline: factory clock replaced natural rhythms; punctuality became a virtue.\n\n"
     "**Long-run effect**: GDP per capita roughly flat for millennia; hockey-stick growth from 1800. Humans today live 40+ years longer than in 1800, with health and material comfort unimaginable to predecessors.",
     9.5, "Q498 Industrial Revolution 1760-1840: steam engine, textiles, iron; Britain first (coal/iron/property rights); urbanisation, initial suffering then rising wages; GDP hockey stick"),

    (499, "Psychology", "What is the placebo effect and how powerful is it really?",
     "Placebo effect: measurable, real physiological or psychological improvement resulting from inert treatment -- driven by expectation, conditioning, and the therapeutic relationship.\n\n"
     "**How real it is**: not 'just in your head' -- measurable biological changes:\n"
     "- Parkinson's patients given saline injections show dopamine release in striatum (de la Fuente-Fernandez 2001)\n"
     "- Placebo painkillers activate endogenous opioid system (naloxone blocks placebo analgesia)\n"
     "- Placebo asthma inhalers improve subjective breathing (though not lung function)\n"
     "- Sham surgery for knee osteoarthritis (Moseley 2002) performed as well as real surgery for pain and function over 2 years\n\n"
     "**Nocebo**: the reverse -- inert treatment causing harm through negative expectation. Patients told a drug causes nausea develop nausea.\n\n"
     "**Factors that increase placebo effect**:\n"
     "- Warm, empathic clinician\n"
     "- Expensive-seeming treatment\n"
     "- More invasive procedure (injection > pill; sham surgery > injection)\n"
     "- Confident delivery ('this will definitely help')\n"
     "- Conditioning (prior positive experience with medication)\n\n"
     "**Open-label placebos**: Kaptchuk (2010) -- patients told explicitly 'these are sugar pills with no active ingredient, but placebos can be powerful.' Significant improvement in IBS. The ritual may be as important as belief.\n\n"
     "**Drug trials**: most psychiatric drugs have placebo response rates of 30-50%. Antidepressant effect sizes over placebo are modest (Kirsch 2008 meta-analysis: ~2 points on Hamilton scale for severe depression). Effect is real but smaller than marketing suggests for mild-moderate cases.\n\n"
     "**Ethical dimension**: deliberately deceiving patients to leverage placebo is ethically fraught. Open-label placebos suggest deception may not even be required.",
     9.0, "Q499 Placebo: real dopamine/opioid release; sham surgery as good as real; nocebo; open-label placebos work; psychiatric drugs 30-50% placebo response; expectation + relationship"),

    (500, "Existentialism", "What did Albert Camus mean by the Absurd, and how should we respond to it?",
     "Camus (1913-1960): one of the 20th century's most important philosophers, though he rejected the label. His central idea: the Absurd.\n\n"
     "**The Absurd**: the conflict between two facts:\n"
     "1. Human beings have an insatiable need for meaning, clarity, and purpose\n"
     "2. The universe offers none -- it is silent, indifferent, chaotic\n"
     "The Absurd is not in humans or the world alone, but in the confrontation between them.\n\n"
     "**The Myth of Sisyphus (1942)**: Sisyphus condemned to roll a boulder up a hill forever, watch it roll back, repeat. Camus: this is the human condition. We work toward goals that ultimately mean nothing.\n\n"
     "**Three responses Camus considered**:\n"
     "1. **Physical suicide**: killing yourself because life is meaningless. Camus rejects this -- it avoids the Absurd rather than confronting it.\n"
     "2. **Philosophical suicide** (Kierkegaard's leap of faith): inventing a god or transcendent meaning to escape the Absurd. Camus calls this intellectual dishonesty -- a 'leap' that denies what reason shows us.\n"
     "3. **Revolt**: Camus's answer. Acknowledge the Absurd fully. Refuse to be crushed by it. Live defiantly in spite of it. Create your own meaning through passionate engagement with life.\n\n"
     "**The conclusion**: 'One must imagine Sisyphus happy.' He owns his fate. The struggle itself toward the heights is enough. Meaning is not found -- it is made, moment by moment, in full knowledge of the Absurd.\n\n"
     "**Camus vs Sartre**: both rejected God, but Sartre found radical freedom empowering; Camus found the silence oppressive but liveable. Camus also rejected Marxist 'leap' -- replacing God with History as a transcendent guarantee.\n\n"
     "**Legacy**: Camus remains the philosopher most read for actually living, not just analysing. The Absurd resonates with modern secular anxiety.",
     10.0, "Q500 Camus Absurd: human need for meaning vs silent universe; Sisyphus happy; 3 responses (suicide/leap/revolt); revolt = defy and create meaning; Camus vs Sartre on freedom"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")