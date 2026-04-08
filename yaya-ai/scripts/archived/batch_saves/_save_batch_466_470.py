"""Save Q466-Q470 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (466, "Biology", "What is epigenetics and can you inherit your parents' experiences?",
     "Epigenetics: changes in gene expression that do not involve changes to the DNA sequence itself. The genome is the hardware; the epigenome is the software that determines which genes are active.\n\n"
     "**Mechanisms**:\n"
     "1. **DNA methylation**: methyl groups attach to cytosine, typically silencing genes. Inherited through cell division.\n"
     "2. **Histone modification**: DNA wraps around histone proteins. Acetylation (relaxes, activates); methylation (condenses, silences). Marks can be heritable.\n"
     "3. **Non-coding RNA**: miRNA and lncRNA regulate gene expression post-transcriptionally.\n\n"
     "**Transgenerational inheritance -- what the evidence shows**:\n"
     "- Dutch Hunger Winter (1944-45): children and grandchildren of famine-exposed women showed metabolic differences -- higher obesity, cardiovascular disease rates decades later.\n"
     "- Swedish Overkalix study: grandfathers' food availability in prepubertal period correlated with grandsons' lifespan.\n"
     "- Mouse studies: stressed/obese parents pass epigenetic marks in sperm/eggs that affect offspring behaviour and metabolism.\n\n"
     "**What it does NOT mean**: you cannot pass on skills or memories (Lamarck was still wrong about that). The effects are statistical tendencies, not deterministic programming.\n\n"
     "**Cancer**: epigenetic dysregulation is a hallmark of cancer. Hypermethylation silences tumour suppressors; hypomethylation activates oncogenes. Epigenetic drugs (HDAC inhibitors) are cancer therapeutics.\n\n"
     "**Reversibility**: unlike genetic mutations, epigenetic marks can be reversed -- by environment, diet, drugs. This is both the hope and complexity of epigenetics.",
     9.0, "Q466 Epigenetics: DNA methylation + histone modification; Dutch Hunger Winter; transgenerational inheritance real but not Lamarckian; cancer connection; reversible"),

    (467, "Physics", "What is the Standard Model of particle physics and what does it leave out?",
     "The Standard Model (SM): our best description of the fundamental particles and forces of nature. Developed 1960s-1970s, confirmed by decades of experiment.\n\n"
     "**Particles**:\n"
     "- Fermions (matter): 6 quarks (up, down, strange, charm, bottom, top) + 6 leptons (electron, muon, tau + their neutrinos)\n"
     "- Bosons (force carriers): photon (EM), W+/W- and Z (weak), gluons x8 (strong), Higgs (mass)\n\n"
     "**Forces in SM** (3 of 4):\n"
     "- Electromagnetic (QED): photons; infinite range; binds atoms\n"
     "- Weak nuclear: W/Z bosons; very short range; radioactive decay, neutrinos\n"
     "- Strong nuclear (QCD): gluons; binds quarks inside protons; confinement\n\n"
     "**Higgs mechanism**: gives mass to W/Z bosons and fermions. Higgs boson confirmed at LHC 2012 (Englert and Higgs, Nobel 2013).\n\n"
     "**What SM leaves out**:\n"
     "1. **Gravity**: no quantum theory of gravity. General relativity breaks at Planck scale. This is the deepest problem in physics.\n"
     "2. **Dark matter**: SM has no candidate particle (WIMPs, axions not in SM)\n"
     "3. **Dark energy**: unexplained\n"
     "4. **Matter-antimatter asymmetry**: SM predicts equal amounts; we're made of matter\n"
     "5. **Neutrino mass**: SM assumes massless; oscillations prove they have mass\n"
     "6. **Hierarchy problem**: why is the Higgs so much lighter than Planck mass?\n\n"
     "**Extensions**: Supersymmetry (no evidence at LHC), string theory (not testable yet), extra dimensions. SM accurate to 1 in 10^12 -- most precise theory ever -- but clearly incomplete.",
     10.0, "Q467 Standard Model: 6 quarks, 6 leptons, force bosons; Higgs confirmed 2012; leaves out gravity, dark matter, matter-antimatter asymmetry, neutrino mass"),

    (468, "Medicine", "How does cancer develop, and why is it so hard to cure?",
     "Cancer: uncontrolled cell division resulting from accumulated mutations that disable normal growth controls.\n\n"
     "**Hallmarks of cancer** (Hanahan & Weinberg 2000, updated 2011):\n"
     "1. Sustaining proliferative signalling\n"
     "2. Evading growth suppressors (p53, Rb)\n"
     "3. Resisting cell death (apoptosis evasion -- Bcl-2 overexpression)\n"
     "4. Enabling replicative immortality (telomerase reactivation)\n"
     "5. Inducing angiogenesis (VEGF -- tumour grows blood supply)\n"
     "6. Activating invasion and metastasis\n"
     "7. Reprogramming energy metabolism (Warburg effect: aerobic glycolysis)\n"
     "8. Evading immune destruction\n\n"
     "**Why mutations accumulate**: DNA replication errors (~1 per 10^9 bp per division); environmental mutagens (UV, tobacco, HPV, aflatoxin); repair system failures (BRCA1/2 = defective double-strand break repair).\n\n"
     "**Why it's hard to cure**:\n"
     "- **Tumour heterogeneity**: one tumour contains millions of genetically distinct cells. Kill most -> survivors resistant.\n"
     "- **Evolution within tumour**: cancer is Darwinian selection under treatment pressure.\n"
     "- **Metastasis**: 90% of cancer deaths from metastases, not primary tumour. Micro-metastases invisible at diagnosis.\n"
     "- **Normal cell similarity**: hard to target cancer without harming normal rapidly dividing cells.\n\n"
     "**Modern advances**: targeted therapy (imatinib for CML -- BCR-ABL inhibitor; remarkable results); immunotherapy (checkpoint inhibitors -- PD-1/CTLA-4 -- 20% of advanced melanoma achieve durable remission); CAR-T for blood cancers; early detection via liquid biopsy.",
     9.5, "Q468 Cancer: Hanahan-Weinberg hallmarks x8; mutations accumulate; heterogeneity + evolution = hard to cure; imatinib, checkpoint inhibitors, CAR-T as breakthroughs"),

    (469, "Philosophy", "What is free will and does it exist?",
     "Free will: the capacity to make choices that are genuinely 'up to you' -- not determined by prior causes.\n\n"
     "**Three positions**:\n"
     "1. **Hard determinism**: every event, including every thought and choice, is causally determined by prior events + laws of physics. Free will is an illusion. (Laplace's demon: perfect knowledge of physics = predict all future.)\n"
     "2. **Libertarian free will** (not political): some choices are genuinely undetermined -- quantum indeterminacy, agent causation, or substance dualism. But random quantum events don't help -- randomness isn't control.\n"
     "3. **Compatibilism** (most philosophers): free will and determinism are compatible. 'Free' means acting from your own desires, values, and reasoning -- not external compulsion. Hume, Frankfurt, Dennett.\n\n"
     "**Neuroscience**: Libet (1983) -- readiness potential precedes conscious awareness of intention by ~500ms. Suggests brain 'decides' before you're aware. Replicated but debated: later studies show you can still 'veto'; timing of awareness measurement disputed.\n\n"
     "**Why it matters**:\n"
     "- **Moral responsibility**: if no free will, can we blame criminals? Compatibilists: yes -- punishment works; people respond to incentives; it's about shaping future behaviour.\n"
     "- **Criminal justice**: some neuroscientists argue we should shift from retributive to rehabilitative justice.\n"
     "- **Psychology**: believing in free will correlates with less cheating, more effort, better outcomes (Baumeister 2008).\n\n"
     "**Consensus**: most philosophers are compatibilists. Hard determinism gains ground in neuroscience. The debate is partly definitional -- what counts as 'free'?",
     9.5, "Q469 Free will: hard determinism vs libertarian vs compatibilism (majority view); Libet 500ms; moral responsibility survives in compatibilism; definitional dispute"),

    (470, "Environmental Science", "What is biodiversity and why does losing it matter?",
     "Biodiversity: the variety of life at three levels -- genetic diversity within species, species diversity within ecosystems, ecosystem diversity across the planet.\n\n"
     "**Current scale**: ~8.7 million eukaryotic species estimated (Mora 2011); only ~2 million described. 1 million threatened with extinction (IPBES 2019). Current extinction rate: 100-1000x background rate -- Earth's 6th mass extinction.\n\n"
     "**Why it matters -- ecosystem services**:\n"
     "1. **Pollination**: 75% of food crops depend on pollinators. Bee decline threatens food security.\n"
     "2. **Pest control**: natural predators suppress agricultural pests. Loss increases pesticide need.\n"
     "3. **Carbon sequestration**: forests, wetlands, soils store carbon. Biodiversity sustains these ecosystems.\n"
     "4. **Water purification**: wetland plants filter pollutants. Mangroves protect coasts from storms.\n"
     "5. **Soil formation**: decomposers, earthworms, mycorrhizae make soil that grows food.\n"
     "6. **Medicine**: 50% of drugs derived from natural compounds. Undiscovered species may hold cures.\n\n"
     "**Stability argument**: diverse ecosystems are more resilient to disturbance. Monocultures collapse easily (Irish Potato Famine: single variety wiped out by single pathogen).\n\n"
     "**Drivers of loss** (HIPPO): Habitat loss (primary), Invasive species, Pollution, Population growth, Overharvesting.\n\n"
     "**Intrinsic value**: many argue species have value independent of human use. The last Northern White Rhino is irreplaceable regardless of economic utility.\n\n"
     "**Solutions**: protected areas (30x30 target); wildlife corridors; rewilding; reducing meat consumption (70% of agricultural land for livestock); international agreements (Kunming-Montreal 2022).",
     9.0, "Q470 Biodiversity: 8.7M species, 1M threatened, 6th mass extinction; ecosystem services (pollination, carbon, medicine); HIPPO drivers; 30x30 target"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
