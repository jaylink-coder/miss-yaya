"""Save Q486-Q490 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (486, "Medicine", "How does the immune system actually work?",
     "The immune system: a layered defence network that distinguishes self from non-self and eliminates pathogens.\n\n"
     "**Two branches**:\n"
     "1. **Innate immunity** (fast, non-specific): first responders. Physical barriers (skin, mucus); phagocytes (neutrophils, macrophages) engulf invaders; natural killer cells destroy infected cells; complement proteins puncture bacteria; inflammation signals danger.\n"
     "2. **Adaptive immunity** (slow, specific, memory): tailored to each threat.\n"
     "   - **B cells**: produce antibodies -- Y-shaped proteins that bind antigens on pathogens. IgG (blood), IgA (gut/lungs), IgE (allergies).\n"
     "   - **T cells**: helper T cells (CD4+) coordinate immune response; cytotoxic T cells (CD8+) kill infected cells directly.\n"
     "   - **Memory cells**: persist after infection -- next exposure triggers faster, stronger response. Basis of vaccination.\n\n"
     "**How it recognises threats**: pattern recognition receptors (Toll-like receptors) detect molecular patterns shared by pathogens (bacterial LPS, viral RNA) -- absent in healthy human cells.\n\n"
     "**MHC (Major Histocompatibility Complex)**: every cell displays peptide fragments on MHC molecules. T cells check these. Foreign peptide = kill. Self peptide = leave alone. MHC mismatch is why transplants are rejected.\n\n"
     "**How it avoids attacking self**: thymic education -- T cells that react to self-antigens are deleted in the thymus. Failures cause autoimmune disease (Type 1 diabetes, lupus, MS, rheumatoid arthritis).\n\n"
     "**Vaccines**: show the immune system a harmless version of a pathogen (live attenuated, killed, subunit, or mRNA-encoded antigen). Memory cells form. Real infection triggers rapid recall response before disease establishes.",
     9.5, "Q486 Immune system: innate (fast/non-specific) + adaptive (B cells/antibodies, T cells, memory); MHC self-recognition; thymic education; autoimmunity when fails; vaccines = memory"),

    (487, "Physics", "What is relativity and why does it matter for everyday technology?",
     "Einstein's two theories of relativity reshaped our understanding of space, time, mass, and energy.\n\n"
     "**Special Relativity (1905)**:\n"
     "- **Two postulates**: (1) laws of physics are the same in all inertial frames; (2) speed of light c is the same for all observers regardless of motion.\n"
     "- **Consequences**: time dilation (moving clocks run slow -- gamma factor = 1/sqrt(1-v^2/c^2)); length contraction; mass-energy equivalence E=mc^2.\n"
     "- **Simultaneity is relative**: two events simultaneous in one frame are not in another moving frame.\n\n"
     "**General Relativity (1915)**:\n"
     "- **Equivalence principle**: gravity and acceleration are locally indistinguishable. A person in a rocket accelerating at g feels identical to standing on Earth.\n"
     "- **Space-time curvature**: mass-energy curves space-time. Objects follow geodesics (straightest paths) in curved space-time -- this IS gravity.\n"
     "- **Predictions**: gravitational time dilation (clocks run slower in stronger gravity); gravitational lensing (light bends around mass); black holes; gravitational waves (confirmed LIGO 2015).\n\n"
     "**Why it matters for everyday technology**:\n"
     "- **GPS**: satellites move fast (SR: clocks slow by 7 microseconds/day) and are farther from Earth's gravity (GR: clocks fast by 45 microseconds/day). Net: +38 microseconds/day. Uncorrected: GPS drifts 10 km/day. Corrected in software -- every GPS device applies relativity.\n"
     "- **Particle accelerators**: relativistic mass increase accounted for in LHC design.\n"
     "- **Nuclear energy**: E=mc^2 -- small mass loss in fission/fusion releases enormous energy.\n"
     "- **PET scans**: positron-electron annihilation produces gamma rays analysed using relativistic kinematics.",
     9.5, "Q487 Relativity: SR (time dilation, E=mc^2, simultaneity) + GR (space-time curvature, gravitational waves); GPS requires both corrections or drifts 10km/day"),

    (488, "Economics", "What is game theory and where does it actually apply?",
     "Game theory: the mathematical study of strategic interactions -- situations where each player's outcome depends on others' choices.\n\n"
     "**Core concepts**:\n"
     "- **Players**: decision-makers with defined strategies and payoffs\n"
     "- **Strategies**: complete plans for every contingency\n"
     "- **Payoffs**: outcomes for each combination of strategies\n"
     "- **Nash Equilibrium**: no player can improve by unilaterally deviating\n\n"
     "**Key games**:\n"
     "- **Prisoner's Dilemma**: individual rationality leads to collectively worse outcome (defect/defect worse than cooperate/cooperate). Explains arms races, overfishing, climate action failure.\n"
     "- **Chicken**: both players prefer to swerve, but each wants the other to swerve. Nuclear deterrence modelled this way.\n"
     "- **Stag Hunt**: cooperation gives best outcome but requires trust. Explains why societies get stuck in low-trust equilibria.\n"
     "- **Repeated games**: folk theorem -- cooperation can emerge if players interact repeatedly and value the future (tit-for-tat in iterated Prisoner's Dilemma).\n\n"
     "**Real applications**:\n"
     "1. **Auctions**: Vickrey auction (second-price sealed bid) makes truthful bidding dominant strategy. FCC spectrum auctions designed using game theory.\n"
     "2. **Nuclear strategy**: MAD -- mutually assured destruction is a Nash equilibrium (neither side benefits from first strike if retaliation guaranteed).\n"
     "3. **Economics**: oligopoly pricing (OPEC), labour bargaining, trade policy.\n"
     "4. **Biology**: evolutionarily stable strategies (Maynard Smith) -- hawk-dove game explains animal conflict.\n"
     "5. **Platform economics**: two-sided markets (Uber, Airbnb) designed using mechanism design.\n\n"
     "**Limits**: assumes rationality; real humans cooperate more than theory predicts (ultimatum game -- people reject unfair offers even at cost to themselves). Behavioural game theory adds psychology.",
     9.0, "Q488 Game theory: Nash equilibrium, Prisoner's Dilemma, repeated games (tit-for-tat); applications in auctions (Vickrey), MAD, OPEC, biology (ESS); real humans more cooperative"),

    (489, "History", "What was the French Revolution and why did it matter?",
     "The French Revolution (1789-1799): the most consequential political upheaval in modern history, dismantling the old order and establishing principles that still define democratic governance.\n\n"
     "**Causes**:\n"
     "1. **Fiscal crisis**: France bankrupt after supporting American Revolution; Louis XVI unable to tax nobility.\n"
     "2. **Enlightenment ideas**: Rousseau, Voltaire, Locke -- popular sovereignty, natural rights, critique of absolute monarchy.\n"
     "3. **Food crisis**: 1788 harvest failure; bread prices soaring; common people starving while aristocracy was exempt from taxes.\n"
     "4. **Social structure**: three estates -- clergy (1st), nobility (2nd), everyone else (3rd, 97% of population). Third Estate paid all taxes, had no political power.\n\n"
     "**Key events**:\n"
     "- 1789: Estates-General convened; Third Estate forms National Assembly; Bastille stormed (July 14); Declaration of Rights of Man\n"
     "- 1791: Constitutional monarchy established\n"
     "- 1792: War with Austria/Prussia; monarchy abolished; First Republic\n"
     "- 1793-94: Reign of Terror (Robespierre, Committee of Public Safety) -- 17,000 executed; 40,000 died in custody\n"
     "- 1795: Thermidorian Reaction -- moderates take power\n"
     "- 1799: Napoleon's coup (18 Brumaire) ends the Revolution\n\n"
     "**Why it mattered**:\n"
     "- Established that sovereignty belongs to the people, not God or kings\n"
     "- Liberty, Equality, Fraternity became template for democratic movements worldwide\n"
     "- Abolished feudalism and privileges of birth overnight\n"
     "- Triggered waves of revolution across Europe (1830, 1848)\n"
     "- Napoleon spread revolutionary legal codes (Code Napoleon) across Europe\n"
     "- Showed revolutions can consume their own children (Robespierre guillotined)",
     9.5, "Q489 French Revolution 1789-99: fiscal crisis + Enlightenment + food shortage; Terror killed 17k; Liberty-Equality-Fraternity template; sovereignty to people; Napoleon result"),

    (490, "Biology", "What is evolution by natural selection and what are the most common misconceptions?",
     "Evolution by natural selection (Darwin 1859, Wallace co-discovery): populations of organisms change over generations because individuals with heritable traits better suited to their environment survive and reproduce more.\n\n"
     "**Four requirements**:\n"
     "1. **Variation**: individuals in population differ\n"
     "2. **Heritability**: variation is passed to offspring\n"
     "3. **Differential reproduction**: some variants reproduce more\n"
     "4. **Time**: accumulate over generations\n\n"
     "**Mechanisms**: natural selection (environment filters); sexual selection (mate choice); genetic drift (random changes, important in small populations); gene flow (migration); mutation (source of new variation).\n\n"
     "**Evidence**: fossil record (transitional forms); comparative anatomy (homologous structures -- bat wing/human hand/whale flipper share bones); molecular evidence (shared DNA, endogenous retroviruses); observed evolution (antibiotic resistance; peppered moths; Galapagos finches beak changes in real time, Grant 1973).\n\n"
     "**Most common misconceptions**:\n"
     "1. 'Evolution is just a theory' -- in science, theory means well-supported explanatory framework (gravity is also 'just a theory')\n"
     "2. 'Organisms evolve on purpose' -- selection has no goal; it is blind. Giraffes didn't try to grow long necks.\n"
     "3. 'Humans evolved from chimps' -- humans and chimps share a common ancestor ~6-7 million years ago.\n"
     "4. 'Evolution means progress' -- no direction or improvement, just fit to current environment. A bacterium is as evolved as a human.\n"
     "5. 'Missing link' -- all fossils are transitional. Tiktaalik (fish with limb-like fins) is one famous example.\n"
     "6. 'It's random' -- mutation is random; selection is not. Selection is the non-random part.\n\n"
     "**Modern synthesis**: Darwinian selection + Mendelian genetics + molecular biology. Supplemented now by evo-devo (development), epigenetics, niche construction.",
     10.0, "Q490 Evolution: 4 requirements (variation/heritability/diff. reproduction/time); mechanisms (selection/drift/flow/mutation); 6 misconceptions; modern synthesis"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
