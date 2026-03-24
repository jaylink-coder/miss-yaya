"""Save Q396-Q400 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (396, "Linguistics", "What is the Sapir-Whorf hypothesis and how strong is the evidence for it?",
     "The strong version (linguistic determinism): language determines thought. Mostly rejected.\n\n"
     "The weak version (linguistic relativity): language *influences* thought. Real evidence:\n\n"
     "**Colour terms**: Russian has goluboy/siniy (light/dark blue). Russian speakers distinguish shades faster near that boundary.\n\n"
     "**Spatial reasoning**: Guugu Yimithirr and Kuuk Thaayorre use absolute cardinal directions (N/S/E/W), not left/right. Speakers outperform English speakers on dead-reckoning.\n\n"
     "**Number**: Pirahã has only 'few' and 'many.' Speakers struggle with exact quantity tasks.\n\n"
     "**Counterfactuals**: Languages with clear counterfactual grammar affect causal reasoning tasks (Au 1983).\n\n"
     "**Conclusion**: you're not trapped by your language — translation works, new concepts can be learned. But lexicalisation makes patterns more salient and automatic. Effect is real but modest.",
     8.5, "Q396 Sapir-Whorf: weak linguistic relativity confirmed (colour, spatial, number); strong determinism rejected"),

    (397, "History", "What caused the collapse of the Bronze Age around 1200 BCE?",
     "One of history's great mysteries: nearly every eastern Mediterranean civilisation collapsed within ~50 years. "
     "Mycenaean Greece, Hittite Empire, Ugarit, Canaan, Egyptian New Kingdom weakened.\n\n"
     "**Proposed causes:**\n"
     "1. **Sea Peoples invasions** — Egyptian records show waves of migrants. Likely a symptom, not cause.\n"
     "2. **Drought/climate change** — Pollen studies and grain records from Ugarit show multi-decade drought from ~1200 BCE. Starvation, state failure, migration, raids.\n"
     "3. **Earthquakes** — Destruction layers at Mycenae, Tiryns. But not all sites.\n"
     "4. **Systems collapse** (Eric Cline, 1177 B.C.): Bronze Age trade was deeply interdependent — tin from Afghanistan, copper from Cyprus, grain from Egypt. One shock cascaded.\n"
     "5. **Internal revolts** — Linear B tablets at Pylos show defensive preparations and reduced rations before destruction.\n\n"
     "**Consensus**: multicausal. Drought stressed food, weakened states, Sea Peoples applied the final blow. Interconnectedness amplified fragility.",
     9.0, "Q397 Bronze Age Collapse ~1200 BCE: multicausal — drought, Sea Peoples, systems collapse (Cline 1177 BC), earthquakes, revolts"),

    (398, "Computer Science", "What is a hash function and why does cryptographic hashing matter?",
     "A hash function maps arbitrary-length input to fixed-length output.\n\n"
     "**Cryptographic hash** requires:\n"
     "1. **Preimage resistance**: given H(x), can't find x.\n"
     "2. **Second preimage resistance**: given x, can't find x' where H(x')=H(x).\n"
     "3. **Collision resistance**: can't find any x, x' where H(x)=H(x').\n"
     "4. **Avalanche effect**: one bit change flips ~50% of output bits.\n\n"
     "**Why it matters:**\n"
     "- Passwords: store H(password), not password. Bcrypt/Argon2 add deliberate slowness.\n"
     "- Digital signatures: sign H(document) — if hash is unforgeable, signature is too.\n"
     "- Integrity: SHA-256 confirms file wasn't tampered with.\n"
     "- Blockchain: each block contains H(previous block) — tamper-evident chain.\n\n"
     "**Broken**: MD5 and SHA-1 have known collisions. Use SHA-256 or SHA-3.",
     9.0, "Q398 Cryptographic hashing: preimage/collision resistance, avalanche effect; SHA-256 ok; MD5/SHA-1 broken"),

    (399, "Philosophy", "What is the problem of induction, and has it been solved?",
     "David Hume (1739): we observe the sun rising every day and conclude it will rise tomorrow. "
     "But what justifies that inference? Past regularities don't logically entail future ones.\n\n"
     "The problem: induction can't be justified by logic (that's deduction) or by past success (circular).\n\n"
     "**Attempted solutions:**\n"
     "1. **Pragmatic** (Reichenbach): if nature is uniform, induction works; if not, no method works. Nothing lost by trying.\n"
     "2. **Falsificationism** (Popper): eliminate induction. Science = bold conjectures + deductive attempts to falsify. Problem: doesn't explain why to trust unfalsified theories.\n"
     "3. **Bayesian**: induction is probabilistic updating. Rational degrees of belief, not certainty. Problem: priors need justification.\n"
     "4. **Dissolution** (Wittgenstein): 'justified belief' just means what it means in practice.\n\n"
     "**Consensus**: not solved. Popper dominant in philosophy of science; Bayesian inference dominates practice. Hume's raw problem remains open.",
     9.0, "Q399 Problem of induction (Hume 1739): can't be justified without circularity; Popper falsification, Bayesian updating, still open"),

    (400, "Biology", "How does CRISPR-Cas9 actually work at the molecular level?",
     "CRISPR is a bacterial immune system repurposed as a gene editor.\n\n"
     "**Mechanism:**\n"
     "1. **Guide RNA (gRNA)**: 20-nucleotide sequence matching target DNA, binds to Cas9.\n"
     "2. **PAM sequence**: Cas9 requires NGG adjacent to target (SpCas9). Without PAM, no cut.\n"
     "3. **Scanning**: gRNA-Cas9 complex scans DNA for complementary sequence next to PAM.\n"
     "4. **Unwinding**: Cas9 unwinds double helix locally for strand invasion.\n"
     "5. **Cleavage**: Two nuclease domains (RuvC + HNH) each cut one strand — blunt double-strand break.\n\n"
     "**Repair pathways:**\n"
     "- **NHEJ**: fast, error-prone; creates indels, disrupts gene. Used to knock out genes.\n"
     "- **HDR**: if template supplied, copies it in for precise editing. Less efficient.\n\n"
     "**Applications**: sickle cell (cured in trials), cancer immunotherapy, crop improvement, diagnostics (SHERLOCK/DETECTR).\n\n"
     "**Safety**: off-target cuts remain a concern. Base editors and prime editors reduce this.",
     9.5, "Q400 CRISPR-Cas9: gRNA+PAM targets, Cas9 cuts both strands; NHEJ=knockout, HDR=precise edit; base/prime editors improving"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
