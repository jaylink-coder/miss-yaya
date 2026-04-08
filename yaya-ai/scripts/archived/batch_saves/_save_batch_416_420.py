"""Save Q416-Q420 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (416, "Psychology", "What is the bystander effect, and when does it NOT apply?",
     "Kitty Genovese (1964) led to Latane and Darley's research. Two mechanisms:\n"
     "1. Diffusion of responsibility: each bystander assumes someone else will act\n"
     "2. Pluralistic ignorance: everyone looks to others, sees calm faces, stays calm\n\n"
     "**Classic experiment** (Darley & Latane 1968): simulated seizure via headphones. Alone: 85% helped within 60s. With 5 'others': 31%. Bystander count reliably reduces intervention.\n\n"
     "**When it does NOT apply:**\n"
     "- Clear emergency with identifiable victim\n"
     "- Bystanders who know each other\n"
     "- Specific responsibility assigned ('You in the red shirt — call 911')\n"
     "- Smaller groups\n"
     "- Simple task with no responsibility diffusion\n\n"
     "**Genovese myth**: Fischer et al. 2011 — 38 witnesses story exaggerated by NYT. Many didn't hear/see clearly. Effect is real, original story was sensationalised.",
     9.0, "Q416 Bystander effect: diffusion of responsibility + pluralistic ignorance; Darley & Latane 1968; overridden by specific assignment, small groups, known bystanders"),

    (417, "Geology", "How do plate tectonics work, and what drives the plates?",
     "~15 major lithospheric plates (100km thick) float on viscous asthenosphere, moving 2-10 cm/year.\n\n"
     "**Three boundary types:**\n"
     "1. Divergent: plates move apart; magma rises; new seafloor created (Mid-Atlantic Ridge, Iceland)\n"
     "2. Convergent: plates collide; oceanic subducts under continental -> trenches (Mariana 11km) + volcanic arcs (Japan, Andes); two continental -> mountain building (Himalayas)\n"
     "3. Transform: plates slide laterally (San Andreas). No creation/destruction of crust.\n\n"
     "**What drives plates (debated):**\n"
     "- Slab pull (strongest, ~90%): cold dense subducting crust pulls plate down by gravity\n"
     "- Ridge push: hot material at ridges creates topographic high pushing plates away\n"
     "- Mantle convection: contribution debated; may be driven by slabs not vice versa\n\n"
     "**Evidence**: magnetic striping on seafloor, continental fit (Wegener), matching fossils, earthquake distributions.",
     9.5, "Q417 Plate tectonics: divergent/convergent/transform boundaries; slab pull main driver (~90%); ridge push + mantle convection also; evidence: magnetic striping, fossils"),

    (418, "Mathematics", "What is a Fourier transform, and why is it everywhere?",
     "Fourier (1822): any periodic function decomposes into sines and cosines of different frequencies.\n\n"
     "Fourier transform: takes time-domain signal f(t) -> frequency-domain F(w) showing how much of each frequency is present.\n"
     "F(w) = integral of f(t)*e^(-iwt) dt. Inverse recovers original.\n\n"
     "**Why it's everywhere:**\n"
     "1. Audio: decompose into frequency spectrum; MP3 compression discards inaudible frequencies\n"
     "2. Images: JPEG uses DCT (related); compression removes high-frequency components\n"
     "3. Differential equations: convolution in time = multiplication in frequency -> algebra\n"
     "4. Quantum mechanics: position and momentum are Fourier duals; Heisenberg uncertainty is a Fourier transform consequence\n"
     "5. MRI: raw data acquired in k-space (frequency); Fourier transform reconstructs image\n"
     "6. Crystallography: X-ray diffraction IS the Fourier transform of electron density\n\n"
     "**FFT** (Cooley-Tukey 1965): O(n log n) vs O(n^2) naive. Made real-time signal processing possible.",
     10.0, "Q418 Fourier transform: time->frequency domain; FFT O(n log n); ubiquitous in audio/images/QM (position-momentum duality)/MRI/crystallography"),

    (419, "Ethics", "What is the difference between deontological and consequentialist ethics, and which is right?",
     "**Consequentialism** (Bentham, Mill): moral worth from consequences. Maximise utility/welfare. Strength: outcome-focused. Weakness: justifies organ harvesting one person to save five; ignores rights.\n\n"
     "**Deontology** (Kant): moral worth from the action itself, not consequences. Categorical Imperative:\n"
     "- Act only by maxims you could universalise\n"
     "- Treat persons as ends, never merely as means\n"
     "Strength: explains why some acts are simply wrong. Weakness: must tell murderer where victim is hiding (can't lie).\n\n"
     "**Virtue ethics** (Aristotle): focus on character. What would a virtuous person do? Eudaimonia as goal.\n\n"
     "**Which is right?** Most philosophers endorse moral pluralism:\n"
     "- Consequentialism: outcomes clearly matter\n"
     "- Deontology: consent and rights matter independently of aggregate utility\n"
     "- Virtue ethics: same act differs morally depending on why it was done\n"
     "Reflective equilibrium: use multiple frameworks as lenses, check coherence with intuitions.",
     9.5, "Q419 Deontology (Kant: actions intrinsically right/wrong) vs consequentialism (outcomes); virtue ethics (Aristotle); most ethicists endorse moral pluralism"),

    (420, "Linguistics", "What is the difference between descriptive and prescriptive grammar, and why does it matter?",
     "**Prescriptive**: what people *should* say — textbooks, style guides, norms ('don't split infinitives', 'it is I').\n\n"
     "**Descriptive**: what people *actually* say — linguist's approach. Records patterns without judging. Every human language has consistent grammar; no language is 'wrong.'\n\n"
     "**Why it matters:**\n"
     "1. Descriptivist rule is fundamental: AAVE has different grammar from SAE, not broken grammar — consistent, learnable patterns\n"
     "2. Prescriptive rules often recent inventions: 'no split infinitives' from 18th-c grammarians applying Latin rules to English incorrectly\n"
     "3. Dialects and class: marking certain dialects as 'incorrect' is a political act correlating with race and class\n"
     "4. But prescriptivism isn't useless: Standard varieties enable communication across dialect lines; style guides serve practical goals\n"
     "5. Code-switching: competent speakers deploy both — AAVE at home, SAE formally. Both correct for their domains.",
     9.0, "Q420 Descriptive vs prescriptive grammar: all languages have systematic grammar; prescriptive rules often arbitrary historical artifacts; dialect marking = political; code-switching"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
