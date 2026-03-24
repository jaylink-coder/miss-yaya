"""Save Q436-Q440 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (436, "Mathematics", "What is a prime number, and what do we know about their distribution?",
     "A prime: natural number > 1 with no divisors other than 1 and itself. Euclid (~300 BCE): infinitely many.\n\n"
     "Fundamental Theorem of Arithmetic: every integer > 1 is prime or unique product of primes.\n\n"
     "**Prime Number Theorem** (Hadamard + de la Vallee Poussin, 1896): pi(x) ~ x/ln(x). Primes become sparser. Better: pi(x) ~ Li(x) = integral of 1/ln(t) from 2 to x.\n\n"
     "**Open problems:**\n"
     "- Twin prime conjecture: infinitely many pairs (p, p+2)? Zhang 2013: proved infinitely many pairs gap < 70M (Polymath reduced to 246)\n"
     "- Goldbach (1742): every even integer > 2 = sum of two primes. Verified to 4*10^18, unproven\n"
     "- Mersenne primes: 2^p - 1. Largest known (2024): 2^136,279,841 - 1 (~41M digits)\n\n"
     "**Practical importance**: RSA encryption relies on difficulty of factoring large numbers.",
     9.0, "Q436 Prime distribution: PNT pi(x)~x/ln(x) (1896); twin primes (Zhang 2013 gap<70M); Goldbach unproven; RSA encryption relies on prime factoring hardness"),

    (437, "Psychology", "How reliable is eyewitness testimony, and what does the research say?",
     "Memory is reconstructive, not reproductive (Loftus). We rebuild memories each time, subject to suggestion.\n\n"
     "**Key experiments:**\n"
     "- Misinformation effect (Loftus 1974): 'smashed vs hit vs contacted' car crash video. 'Smashed' group: 32% reported broken glass that wasn't there.\n"
     "- Lost in the mall (1995): planted false memory of childhood event. ~25% came to believe and elaborate it.\n\n"
     "**Courtroom impact**: Innocence Project DNA exonerations — eyewitness misidentification in ~70% of wrongful convictions. Witness confidence is NOT correlated with accuracy.\n\n"
     "**Why malleable**: each recall reconstructs using schemas, expectations, emotions; post-event information integrates into memory.\n\n"
     "**What helps**: sequential lineups, blind administration, record initial statement, open questions first.",
     9.5, "Q437 Eyewitness memory (Loftus): reconstructive not reproductive; misinformation effect (smashed vs hit); 70% wrongful convictions involve misidentification; confidence unreliable"),

    (438, "Physics", "What is the difference between heat and temperature?",
     "**Temperature**: average kinetic energy per molecule. Measured in Kelvin/Celsius/Fahrenheit.\n\n"
     "**Heat**: *transfer* of thermal energy between objects due to temperature difference. Heat is a process, not a property. Pool at same temperature as coffee mug: pool contains far more thermal energy.\n\n"
     "**Specific heat capacity**: energy per kg per degree change. Water: ~4,200 J/kg*K (high — moderates climate). Iron: ~450 J/kg*K (heats/cools fast).\n\n"
     "**Zeroth Law**: if A equilibrates with B, and B with C, then A with C.\n\n"
     "**Why metal feels colder than wood at same temperature**: metal conducts heat away faster (high thermal conductivity). Same temperature, different heat flow.\n\n"
     "**Water vs air at 100C**: water burns much more — higher thermal mass + latent heat of condensation deliver more energy per contact.",
     9.0, "Q438 Heat vs temperature: temperature = average KE/molecule; heat = energy transfer; water high specific heat (4200 J/kg*K) moderates climate; metal feels cold = conducts heat away"),

    (439, "Sociology", "What is structural racism, and how does it differ from individual racism?",
     "**Individual racism**: prejudice and discriminatory acts by identifiable individuals. Overt bias.\n\n"
     "**Structural racism**: racial disparities from systems, policies, norms — even without individual racist intent.\n\n"
     "**Examples:**\n"
     "1. Redlining (1930s HOLC): Black neighbourhoods marked hazardous, denied mortgages. Compounding disinvestment; formerly redlined areas still show lower values, worse schools, higher poverty 90 years later.\n"
     "2. Crack/powder cocaine sentencing (100:1 ratio to 2010): identical drug, racially disparate impact, no individual intent needed.\n"
     "3. Hiring algorithms: Amazon's CV screener (abandoned 2018) downranked women — trained on biased historical data.\n"
     "4. Wealth gap: Black families ~8% of white family net worth — compounded effect of slavery, Jim Crow, redlining.\n\n"
     "**Evidence**: audit studies (identical CVs with Black vs white names) consistently find measurable hiring discrimination.",
     9.0, "Q439 Structural racism vs individual racism: systemic disparities without individual intent; redlining, sentencing disparities, algorithm bias, wealth gap; audit studies confirm"),

    (440, "Ecology", "What is the nitrogen cycle and why is it essential for life?",
     "N2 = 78% of atmosphere, but most organisms can't use N2 (triple bond N=N, 945 kJ/mol). Life needs reactive N (NH3, NO3-, organic N).\n\n"
     "**Nitrogen cycle:**\n"
     "1. Fixation: N2 -> NH3 by free-living bacteria (Azotobacter), symbiotic bacteria in legume roots (Rhizobium), lightning, and Haber-Bosch process (N2 + H2 -> NH3, high T/pressure/Fe; feeds ~50% of humanity)\n"
     "2. Nitrification: NH3 -> NO2- -> NO3- (Nitrosomonas, Nitrobacter)\n"
     "3. Assimilation: plants absorb NO3- or NH4+, incorporate into amino acids/proteins/DNA\n"
     "4. Ammonification: decomposers break organic N -> NH3\n"
     "5. Denitrification: bacteria (Pseudomonas) convert NO3- -> N2 in anaerobic conditions\n\n"
     "**Limiting nutrient**: N limits most ecosystems. Excess (fertilisers) -> eutrophication -> algal blooms -> oxygen depletion -> dead zones (Gulf of Mexico). Haber also weaponised N (chlorine, phosgene).",
     9.0, "Q440 Nitrogen cycle: N2 (triple bond) fixed by Rhizobium + Haber-Bosch (feeds 50% of humanity); nitrification, assimilation, denitrification; excess N = eutrophication"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total memories: {total}")
