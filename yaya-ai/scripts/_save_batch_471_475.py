"""Save Q471-Q475 batch."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.save_qa_entry import save_qa

entries = [
    (471, "Neuroscience", "How does addiction work in the brain, and why is it so hard to quit?",
     "Addiction: compulsive substance use or behaviour despite harmful consequences. A brain disease, not a moral failing.\n\n"
     "**The dopamine system**: the mesolimbic pathway (VTA to nucleus accumbens) is the core reward circuit. Normally releases dopamine in response to natural rewards (food, sex, social connection). Drugs hijack this system -- cocaine blocks dopamine reuptake (10x normal spike); heroin mimics endorphins; alcohol enhances GABA and suppresses glutamate.\n\n"
     "**Neuroadaptation**:\n"
     "1. **Tolerance**: repeated stimulation downregulates dopamine receptors -- need more drug for same effect\n"
     "2. **Withdrawal**: baseline dopamine drops below normal -- dysphoria, craving, physical symptoms\n"
     "3. **Sensitisation**: drug cues (environment, paraphernalia) trigger dopamine release even without drug -- Pavlovian conditioning. This persists for years.\n\n"
     "**Prefrontal cortex damage**: addiction impairs the PFC, which controls impulse suppression. Less top-down control over limbic drives. Decision-making becomes biased toward immediate reward.\n\n"
     "**Why quitting is hard**:\n"
     "- Withdrawal is acutely miserable\n"
     "- Cue-triggered cravings persist for years (rat park experiments: environment matters enormously)\n"
     "- Social networks often revolve around substance\n"
     "- Underlying trauma/mental illness driving use\n\n"
     "**Treatment**: best evidence for opioid addiction -- methadone and buprenorphine (harm reduction). Naltrexone for alcohol/opioids. CBT, motivational interviewing. Peer support (AA: mixed evidence but social component powerful). No magic cure -- relapse rates ~40-60%.",
     9.5, "Q471 Addiction: dopamine hijack (VTA-accumbens); tolerance+withdrawal+sensitisation; PFC impairment; cue-conditioning persists years; methadone/buprenorphine best evidence"),

    (472, "Mathematics", "What is a prime number and why do mathematicians care so deeply about them?",
     "A prime number is a natural number greater than 1 with no positive divisors other than 1 and itself. First primes: 2, 3, 5, 7, 11, 13...\n\n"
     "**Fundamental Theorem of Arithmetic**: every integer > 1 is either prime or uniquely factorisable into primes. Primes are the atoms of number theory.\n\n"
     "**Distribution**: primes become sparser as numbers grow, but never stop (Euclid's proof: assume finitely many, multiply them all and add 1 -- the result is divisible by none of them, contradiction). Prime Number Theorem: primes near N appear with density approximately 1/ln(N).\n\n"
     "**Twin primes**: pairs differing by 2 (3,5; 11,13; 17,19). Conjectured to be infinite -- unproven. Zhang 2013: proved gaps < 70 million infinitely often (later reduced to 246 by Maynard).\n\n"
     "**Why mathematicians care**:\n"
     "1. **RSA cryptography**: security rests on the difficulty of factoring large semiprimes (p x q). Factor a 2048-bit number and you break internet encryption.\n"
     "2. **Riemann Hypothesis**: zeros of the zeta function control prime distribution. Still unproven.\n"
     "3. **Addbach's conjecture**: every even integer > 2 is the sum of two primes (e.g. 4=2+2, 28=5+23). Verified to 4x10^18. Unproven.\n"
     "4. **Mersenne primes**: of form 2^p - 1. Used in computing, random number generation. Only 51 known.\n\n"
     "**Largest known prime** (as of 2024): 2^136,279,841 - 1 (over 41 million digits). Found by GIMPS distributed computing project.\n\n"
     "**Practical**: AES, RSA, HTTPS, Bitcoin -- all depend on prime-based mathematics.",
     9.0, "Q472 Primes: atoms of arithmetic (FTA); Euclid's infinity proof; RSA cryptography; Goldbach conjecture; Riemann zeta connection; largest known 41M digits"),

    (473, "Psychology", "What is trauma and how does it rewire the brain?",
     "Trauma: an overwhelming experience that exceeds the capacity to cope, leaving lasting psychological and neurobiological changes.\n\n"
     "**PTSD neurobiology**:\n"
     "1. **Amygdala** (fear centre): hyperactivated in PTSD -- threat detection system stuck on high alert. Responds to cues associated with trauma even in safety.\n"
     "2. **Hippocampus** (memory): volume reduced in PTSD (~8% smaller on average). Impairs contextualisation of fear memories -- trauma feels present, not past.\n"
     "3. **Prefrontal cortex** (regulation): hypoactivated -- reduced capacity to 'talk down' the amygdala. Explains intrusive thoughts, hypervigilance.\n"
     "4. **HPA axis**: stress hormone system (cortisol) dysregulated -- either chronically elevated or blunted (as seen in combat veterans).\n\n"
     "**Polyvagal theory** (Porges): trauma hijacks the autonomic nervous system -- freeze, fight, flight. Ventral vagal (social engagement) offline; dorsal vagal (shutdown) engaged.\n\n"
     "**Developmental trauma**: ACEs (Adverse Childhood Experiences) study (Felitti 1998) -- 10 categories of childhood adversity. 4+ ACEs dramatically increase rates of depression, addiction, obesity, heart disease, early death. Dose-response relationship.\n\n"
     "**Treatment**:\n"
     "- EMDR (Eye Movement Desensitisation and Reprocessing): evidence-based, possibly reconsolidates traumatic memories\n"
     "- Prolonged Exposure: gradual desensitisation\n"
     "- CPT (Cognitive Processing Therapy): restructures trauma-related beliefs\n"
     "- Ketamine and MDMA-assisted therapy: promising Phase 3 trials for PTSD\n\n"
     "**Resilience**: most people exposed to trauma do NOT develop PTSD. Social support is the strongest protective factor.",
     9.5, "Q473 Trauma/PTSD: amygdala hyperactive, hippocampus shrunken, PFC offline; HPA dysregulation; ACEs dose-response; EMDR+CPT+MDMA trials; social support protective"),

    (474, "Economics", "What is inflation, what causes it, and how is it controlled?",
     "Inflation: a general, sustained rise in the price level -- equivalently, a decline in purchasing power of money.\n\n"
     "**Measurement**: CPI (Consumer Price Index) -- basket of goods/services; PCE (Personal Consumption Expenditures) -- Fed's preferred measure; Core (excludes food and energy -- more stable).\n\n"
     "**Causes**:\n"
     "1. **Demand-pull**: too much money chasing too few goods. Economy overheating; spending exceeds productive capacity.\n"
     "2. **Cost-push**: supply shocks raise production costs (oil crises 1973, 2022 energy shock) -- firms pass costs to consumers.\n"
     "3. **Built-in (wage-price spiral)**: workers demand higher wages to compensate for inflation; firms raise prices to cover wage costs; repeat.\n"
     "4. **Monetary**: Milton Friedman: 'Inflation is always and everywhere a monetary phenomenon.' Excessive money supply growth (printing money) devalues currency.\n\n"
     "**Hyperinflation examples**: Weimar Germany 1923 (paper money as wallpaper); Zimbabwe 2008 (100 trillion dollar note); Venezuela 2018 (1,000,000% annual rate).\n\n"
     "**Control -- central bank tools**:\n"
     "- **Interest rates**: raise rates -> credit more expensive -> less borrowing/spending -> demand falls -> prices stabilise. 2022-2023: Fed raised rates from 0.25% to 5.25% to fight post-COVID inflation.\n"
     "- **Quantitative tightening**: reduce money supply by selling assets.\n"
     "- **Reserve requirements**: how much banks must hold vs lend.\n\n"
     "**Trade-off**: fighting inflation slows growth, raises unemployment (Phillips Curve -- inverse relationship, though contested post-1970s stagflation).\n\n"
     "**Target**: most central banks target 2% inflation -- enough to avoid deflation, small enough to not distort decisions.",
     9.0, "Q474 Inflation: demand-pull, cost-push, wage-price spiral, monetary (Friedman); CPI/PCE measurement; central bank interest rates; 2% target; Phillips Curve trade-off"),

    (475, "Philosophy", "What is the meaning of life -- and is it even a well-formed question?",
     "Possibly the oldest philosophical question. The answer depends on what 'meaning' means.\n\n"
     "**Three sub-questions** (Susan Wolf):\n"
     "1. What is the meaning of life? (cosmic purpose)\n"
     "2. What is a meaningful life? (personal significance)\n"
     "3. What makes life worth living? (value, wellbeing)\n\n"
     "**Major answers**:\n\n"
     "**Religious**: life has meaning given by a creator. Purpose is to fulfil divine will, achieve salvation, or realise dharma. Meaning is objective and given, not constructed.\n\n"
     "**Nihilism** (Nietzsche's diagnosis, not prescription): God is dead; with Him, objective meaning is gone. The universe is indifferent. Nothing matters inherently.\n\n"
     "**Existentialism** (Sartre, Camus): existence precedes essence -- we have no preset purpose. We create meaning through authentic choices. Camus: life is absurd (we crave meaning in an indifferent universe); the response is rebellion, not despair.\n\n"
     "**Analytic approaches**:\n"
     "- **Desire satisfaction**: life is meaningful when you get what you want. Problem: you can want trivial or harmful things.\n"
     "- **Objective list** (Parfit): knowledge, friendship, achievement, aesthetic experience. Meaningful regardless of whether desired.\n"
     "- **Engagement theory** (Wolf): meaning arises from active engagement with objectively worthwhile projects.\n\n"
     "**Is it well-formed?** Wittgenstein: 'The solution to the problem of life is seen in the vanishing of the problem.' The question may dissolve under analysis -- not answerable because it's not asking for a specific fact.\n\n"
     "**Working answer** most find satisfying: meaning is constructed, not found. Deep relationships, purposeful work, growth, contribution to something beyond yourself. Consistent across cultures and secular/religious traditions.",
     9.0, "Q475 Meaning of life: 3 sub-questions (Wolf); religious/nihilist/existentialist/analytic approaches; Camus absurdism; Wittgenstein dissolution; meaning constructed not found"),
]

for q_num, domain, question, answer, rating, summary in entries:
    total = save_qa(q_num, domain, question, answer, rating, summary)
    print(f"Q{q_num} ({domain}) saved. Total: {total}")
