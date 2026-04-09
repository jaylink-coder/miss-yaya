"""Save Q501-Q505 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (501, "Economics", "What is inflation, and why is it so hard to control?",
     "Inflation: sustained rise in the general price level — or equivalently, fall in purchasing power of money. If prices rise 7%, your salary buys 7% less.\n\n"
     "**What causes it**:\n"
     "1. **Demand-pull**: too much money chasing too few goods. Wartime economies; massive stimulus.\n"
     "2. **Cost-push**: supply costs rise (oil shock 1973; COVID supply chains). Costs passed to prices.\n"
     "3. **Built-in (wage-price spiral)**: workers expect inflation → demand higher wages → firms raise prices to compensate → repeat.\n"
     "4. **Monetary**: Milton Friedman: 'Inflation is always and everywhere a monetary phenomenon.' Print more money than output grows → prices rise. Weimar Germany 1923; Zimbabwe 2008 (100 trillion dollar note).\n\n"
     "**The Fisher Equation**: MV = PQ. Money supply × velocity = price level × real output. Hold V and Q constant: doubling M doubles P.\n\n"
     "**Why hard to control**:\n"
     "- Central banks use interest rates (raise rates → borrowing costly → demand falls → prices stabilise). But:\n"
     "- Rate rises hurt debtors, raise unemployment, can trigger recessions.\n"
     "- Supply-side inflation (energy, food) doesn't respond to rate hikes — you're crushing demand to fight a supply problem.\n"
     "- Expectations are self-fulfilling: if people believe 10% inflation is coming, they demand 10% wage rises, making it happen.\n"
     "- Political pressure: governments don't want high rates near elections.\n\n"
     "**The 2% target**: most central banks target 2% — low enough to preserve purchasing power; high enough to avoid deflation (falling prices → consumers defer spending → recession death spiral; Japan's 'lost decades').\n\n"
     "**2021-2023 inflation**: COVID stimulus + supply chain disruption + energy shock (Ukraine war) combined all three causes simultaneously. Fed raised rates 500bps in 18 months — fastest since Volcker 1980.",
     9.0, "Q501 Inflation: demand-pull/cost-push/monetary causes; MV=PQ; 2% target avoids deflation; rate hikes blunt instrument; 2021 inflation = stimulus+supply+energy combined"),

    (502, "Genetics", "What is CRISPR-Cas9 and how does gene editing work?",
     "CRISPR-Cas9: a molecular tool that lets scientists edit DNA with unprecedented precision. Nobel Prize 2020 (Doudna and Charpentier). Derived from a bacterial immune system.\n\n"
     "**How bacteria use CRISPR**: when a virus infects a bacterium, the bacterium can store fragments of viral DNA in its genome (CRISPR arrays). If the same virus attacks again, the bacterium transcribes these into guide RNA, which directs the Cas9 protein to cut the viral DNA. It is a biological immune memory.\n\n"
     "**How scientists repurposed it**:\n"
     "1. Design a guide RNA (gRNA) that matches any target DNA sequence (20 bases)\n"
     "2. Load it into Cas9 protein\n"
     "3. Deliver to cells (virus vector, lipid nanoparticle, electroporation)\n"
     "4. Cas9 scans DNA, finds the matching sequence + PAM motif (NGG), cuts both strands\n"
     "5. Cell's repair machinery kicks in:\n"
     "   - **NHEJ** (non-homologous end joining): error-prone repair → insertions/deletions → gene knockout\n"
     "   - **HDR** (homology-directed repair): provide a template → precise edit or gene insertion\n\n"
     "**Applications**:\n"
     "- **Medicine**: sickle cell disease and beta-thalassemia cured in trials (2023) — edit stem cells to reactivate fetal haemoglobin. Cancer immunotherapy (edit T cells). Inherited blindness.\n"
     "- **Agriculture**: disease-resistant crops; hornless cattle; non-browning mushrooms.\n"
     "- **Research**: knock out any gene to study its function.\n\n"
     "**Ethical concerns**:\n"
     "- **Germline editing**: He Jiankui (2018) edited human embryos to resist HIV — twins born, scientist jailed. Heritable changes affect all descendants — irreversible.\n"
     "- **Off-target cuts**: Cas9 occasionally cuts wrong sites.\n"
     "- **Enhancement vs therapy**: where is the line?\n\n"
     "**Next generation**: base editors (change one nucleotide without cutting); prime editors ('find and replace' for DNA). More precise, fewer off-targets.",
     9.5, "Q502 CRISPR-Cas9: bacterial immune system repurposed; guide RNA directs Cas9 to cut DNA; NHEJ=knockout, HDR=precise edit; sickle cell cured 2023; He Jiankui germline scandal; base editors next"),

    (503, "Philosophy", "Does the self exist — or is personal identity an illusion?",
     "The self: the sense of being a continuous, unified 'I' persisting through time. One of philosophy's hardest questions — and neuroscience is making it harder.\n\n"
     "**The bundle theory (Hume 1739)**: look inward for a 'self' — you find only a stream of perceptions: thoughts, feelings, sensations, memories. No observer behind them. The self is a bundle of experiences, not a separate thing. Like a nation: many components, no single 'nationhood' beyond them.\n\n"
     "**The narrative self (Dennett, Ricoeur)**: the self is a story the brain tells — a coherent narrative constructed from experience. Not a fixed entity but an ongoing fiction. Useful fiction, but fiction nonetheless. The brain's 'press secretary' rationalising decisions already made.\n\n"
     "**Personal identity over time (Locke vs Parfit)**:\n"
     "- Locke: identity = psychological continuity — memory connects your past self to present self.\n"
     "- Problem: if teleporter destroys you and recreates an identical person on Mars, is that you?\n"
     "- Parfit (1984): personal identity is not what matters. What matters is psychological continuity and connectedness. If the Mars copy has all your memories and character, the question 'is it really you?' may be empty. Identity is not a deep fact.\n\n"
     "**The neuroscience view**: no single 'self centre' in the brain. Default mode network generates self-referential processing but is distributed. Split-brain patients (corpus callosum severed) show two semi-independent streams of consciousness — one 'self' or two?\n\n"
     "**Buddhist philosophy**: anatta (no-self) — the sense of a permanent, unified self is a cognitive illusion that causes suffering. Meditation reveals the constructed nature of self.\n\n"
     "**The paradox**: whatever asks 'does the self exist?' is itself the self. The question may be self-refuting — or self-illuminating.",
     9.0, "Q503 Self/personal identity: Hume bundle theory; Dennett narrative self; Parfit psychological continuity; personal identity not what matters; neuroscience no self-centre; Buddhist anatta"),

    (504, "Astrophysics", "What is a black hole — and what actually happens inside one?",
     "Black hole: a region of spacetime where gravity is so extreme that nothing — not even light — can escape past the event horizon. Not a hole; a massive, collapsed object.\n\n"
     "**Formation**: when a massive star (>20 solar masses) exhausts its nuclear fuel, radiation pressure fails, core collapses under gravity, rebounds in supernova, leaving behind a singularity. Supermassive black holes (millions-billions of solar masses) at galactic centres: formation unclear — primordial? Grown by accretion and mergers?\n\n"
     "**Key features**:\n"
     "- **Event horizon**: the point of no return. Radius = Schwarzschild radius: r = 2GM/c². For Sun: ~3km. For Earth: ~9mm.\n"
     "- **Singularity**: at the centre, density infinite, spacetime curvature infinite. GR breaks down here — quantum gravity needed.\n"
     "- **Hawking radiation** (1974): quantum effects near event horizon cause black holes to slowly emit thermal radiation and eventually evaporate. Temperature inversely proportional to mass. Stellar-mass BH: evaporation time >> age of universe.\n\n"
     "**What happens if you fall in** (from your perspective):\n"
     "- For a large BH: cross event horizon without drama — tidal forces modest at r_s.\n"
     "- No local 'wall' — you feel nothing special at the horizon.\n"
     "- Tidal forces (spaghettification) kill you as you approach singularity.\n"
     "- Light from outside universe appears increasingly blueshifted.\n\n"
     "**From outside observer's perspective**:\n"
     "- You appear to slow down, freeze, redshift to invisibility at horizon.\n"
     "- You never appear to cross — time dilation becomes infinite at event horizon.\n\n"
     "**The information paradox**: Hawking radiation is thermal (random) — so when a BH evaporates, all information about what fell in is destroyed. But quantum mechanics says information cannot be destroyed. Unresolved. Possible resolution: holographic principle — information encoded on event horizon surface.\n\n"
     "**Confirmed observations**: EHT imaged M87* (2019) and SgrA* (2022) — first direct images. Gravitational waves from BH mergers detected by LIGO (2015).",
     9.5, "Q504 Black holes: formation from stellar collapse; event horizon r=2GM/c²; singularity; Hawking radiation; spaghettification; outside observer sees freeze; information paradox; EHT imaged M87* 2019"),

    (505, "Linguistics", "How does language shape thought — does your language change how you think?",
     "Linguistic relativity (Sapir-Whorf hypothesis): the language you speak influences how you perceive and think about the world. Two versions: strong (language determines thought — now rejected) and weak (language influences thought — supported by evidence).\n\n"
     "**The strong hypothesis (Whorf 1940s)**: argued Hopi Indians had no tense — therefore thought about time differently. Later debunked: Hopi does have temporal concepts; Whorf's analysis was flawed.\n\n"
     "**Modern evidence for weak version** (Boroditsky et al.):\n"
     "- **Colour perception**: Russian has distinct words for light blue (goluboy) and dark blue (siniy). Russian speakers discriminate shades of blue faster than English speakers in the blue-boundary range. Language primes categorical perception.\n"
     "- **Spatial orientation**: Guugu Yimithirr (Australia) uses cardinal directions (north/south/east/west) not relative (left/right/behind). Speakers maintain perfect orientation at all times, even indoors. Language shapes navigation.\n"
     "- **Number**: Piraha (Amazon) has no exact number words above 2. Speakers cannot reliably perform exact numerical tasks.\n"
     "- **Gender**: Spanish/French nouns have gender. Speakers describe gendered objects using attributes matching grammatical gender (bridge = masculine in German, feminine in Spanish → described differently).\n"
     "- **Time direction**: Mandarin speakers more naturally think of time vertically (earlier above, later below). English speakers think horizontally.\n\n"
     "**What language doesn't determine**: pre-linguistic infants and great apes have numerical, spatial, and causal concepts without language. Thought precedes language developmentally; language is not the sole medium of thought.\n\n"
     "**Practical implication**: multilingual people report feeling 'different' in different languages — different emotional tone, different social identity, different association networks. Language is not a cage but a lens.",
     9.0, "Q505 Linguistic relativity: Sapir-Whorf; strong version rejected; weak version supported; Russian blue discrimination; Guugu spatial orientation; Piraha numbers; Mandarin vertical time; language as lens not cage"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")