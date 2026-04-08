"""Save Q421-Q425 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (421, "Biology", "How does the immune system distinguish 'self' from 'non-self'?",
     "Multiple overlapping mechanisms:\n\n"
     "**MHC class I**: every nucleated cell displays own protein fragments on MHC class I. T-cells educated in thymus to tolerate self-peptides; cells displaying foreign peptides (viral, tumour) are destroyed.\n\n"
     "**Central tolerance** (thymus): T-cells reacting too strongly to self-antigens deleted (negative selection, ~95% die). Useful cells survive (positive selection).\n\n"
     "**Peripheral tolerance**: regulatory T-cells (Tregs) suppress self-reactive escapees. Anergy: T-cells encountering antigen without co-stimulatory signals become unresponsive.\n\n"
     "**Innate immune system**: pattern recognition receptors (TLRs, NLRs) detect PAMPs (pathogen-associated molecular patterns) — bacterial LPS, viral dsRNA, fungal cell wall — which self-cells don't have.\n\n"
     "**When it fails:**\n"
     "- Autoimmunity (MS, RA, T1D): molecular mimicry or tolerance breakdown\n"
     "- Transplant rejection: donor MHC recognised as foreign",
     9.5, "Q421 Immune self/non-self: MHC class I display, thymic selection (95% die), Tregs, PAMPs; fails in autoimmunity (molecular mimicry) and transplant rejection"),

    (422, "History", "What was the significance of the printing press, and did Gutenberg invent it?",
     "Gutenberg (~1450) didn't invent printing. Moveable type existed in China (Bi Sheng ~1040) and Korea (Jikji 1377, oldest surviving). Gutenberg's innovation: complete system — durable metal alloy type, oil-based ink, adapted screw press — optimised for the European alphabet.\n\n"
     "**Why it transformed Europe**: Latin alphabet (26 letters) made moveable type far more efficient than Chinese/Korean systems (thousands of characters). Press: ~3,600 pages/day vs 40 for a scribe.\n\n"
     "**Impact:**\n"
     "- Bibles: ~180 Gutenberg Bibles (~1455); scripture beyond monasteries\n"
     "- Reformation: Luther's 95 Theses spread Germany in 2 weeks; 300,000 pamphlet copies by 1520\n"
     "- Scientific Revolution: Copernicus, Galileo, Newton published to pan-European audiences; cumulative science enabled\n"
     "- Vernacular languages: publishing in German/French/English accelerated nationalisms; Latin dominance ended\n"
     "- Book prices fell 80% in 50 years; literacy expanded to middle class\n\n"
     "**Eisenstein's thesis**: press created new relationship with knowledge — standardisation, reproducibility, verification.",
     9.5, "Q422 Gutenberg press (~1450): moveable type existed in China/Korea first; transformed Europe via Reformation, Scientific Revolution, vernacular languages; Eisenstein: new knowledge relationship"),

    (423, "Neuroscience", "What is consciousness and why is it so hard to explain?",
     "Chalmers (1995) 'hard problem': why does information processing produce *subjective experience*? Why is there something it is like to see red?\n\n"
     "Even full neural mapping of correlates wouldn't explain why any of it feels like anything. Explaining function doesn't explain qualia.\n\n"
     "**Major theories:**\n"
     "1. Global Workspace Theory (Baars, Dehaene): consciousness = information broadcast across brain; frontoparietal involvement; supported by fMRI/MEG\n"
     "2. Integrated Information Theory (Tononi): consciousness = Phi (integrated information). High Phi = conscious. Controversial: XOR grids have high Phi.\n"
     "3. Higher-Order Theories: conscious of X only if higher-order representation of X (thought about thought); prefrontal involvement\n"
     "4. Predictive Processing (Friston): consciousness = brain's model of itself; perception = prediction\n"
     "5. Illusionism (Frankish): qualia don't exist as intuited — introspective reports are systematically wrong\n\n"
     "**Consensus**: none. Hard problem remains genuinely open.",
     10.0, "Q423 Consciousness hard problem (Chalmers 1995): why experience? GWT (broadcast), IIT (Phi), higher-order, predictive processing, illusionism; no consensus"),

    (424, "Chemistry", "What makes some molecules smell while others don't?",
     "Odorant molecules bind to olfactory receptors (ORs, GPCRs) in nasal epithelium -> olfactory bulb -> piriform cortex + limbic system.\n\n"
     "**What molecules smell:**\n"
     "1. Volatility: must be airborne; large molecules don't vaporise; <300 Da typical\n"
     "2. Lipophilicity: must dissolve in mucus AND bind to OR proteins. Too hydrophilic or hydrophobic = no smell\n"
     "3. Molecular shape: OR binding pocket specific; carvone enantiomers: L = spearmint, D = caraway — same formula, opposite smell\n\n"
     "**Olfactory code**: humans have ~400 functional OR genes. Each OR responds to range of odorants. Each odorant activates a pattern — combinatorial code like chords.\n\n"
     "**Concentration extremes**: thiols (skunk, garlic) detectable at 1 ppb; geosmin (petrichor) at 5 ppt — high receptor affinity.\n\n"
     "**COVID-19 anosmia**: virus destroys olfactory neuron support cells; usually recovers.",
     9.0, "Q424 Olfaction: volatility + lipophilicity + shape -> OR binding (400 GPCR types); combinatorial code; carvone enantiomers have opposite smells; COVID anosmia"),

    (425, "Economics", "What caused the 2008 financial crisis?",
     "Cascade of failures:\n\n"
     "**Housing bubble**: low post-9/11 rates, relaxed lending, NINJA loans (no income/assets/job). Prices +124% nationally 2000-2006.\n\n"
     "**Securitisation**: banks sold mortgages as MBS, removing incentive to evaluate creditworthiness. Originate-to-distribute model.\n\n"
     "**CDOs**: MBS re-bundled, given AAA ratings by agencies (Moody's, S&P) with conflict of interest — paid by issuers.\n\n"
     "**Leverage**: Lehman Brothers 30:1 leverage ($600B assets, $22B equity). 3% asset decline = equity wiped out. AIG wrote $400B+ of CDS with no reserves.\n\n"
     "**Trigger**: house prices peaked 2006, fell. Defaults rose. MBS values fell. Interbank lending froze. Lehman collapsed September 15, 2008. AIG needed $182B bailout.\n\n"
     "**Result**: $19T US household wealth lost. 8.7M jobs lost. Global recession. TARP $700B, rates to zero.\n\n"
     "**Response**: Dodd-Frank, Basel III capital requirements, Volcker Rule.",
     9.5, "Q425 2008 crisis: housing bubble + NINJA loans + securitisation (originate-distribute) + CDO rating fraud + 30:1 leverage; Lehman collapse; $19T wealth lost; Dodd-Frank"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
