# Queueing Theory Thesaurus

[🇷🇺 Русская версия](tesaurus.ru.md)

Below is a structured "thesaurus" of queueing theory terminology.

---

## 1. Basic Elements of a Queueing System

- **Queueing system**  
  An abstract model of a real service system (checkout counters, servers, communication lines, etc.) in which:
  - jobs (customers) arrive,
  - waiting in a queue is possible,
  - service is performed by one or more channels.

- **Job / customer / call / client**  
  An entity that arrives at the system and requires service (a customer in a bank, a packet in a network, a call at a telephone exchange, etc.).

- **Source of jobs**  
  The population (real or abstract) from which jobs arrive. It may be:
  - **unlimited** (a potentially infinite number of customers),
  - **limited** (a fixed number of customers, e.g., $N$ subscribers).

- **Service channel (server)**  
  The element of the queueing system that directly performs the service of one (sometimes several) jobs.

- **Number of service channels $c$**  
  The number of parallel servers in the system.

- **Queue**  
  The set of jobs waiting for service to begin.

- **Buffer / system capacity**  
  The maximum number of jobs that can be in the system at the same time (in service + in the queue). Denoted by $K$.

- **Waiting (delay) system**  
  A queueing system where jobs may wait in the queue when all channels are busy.

- **Loss system**  
  A queueing system where, when all channels are busy, arriving jobs are immediately rejected and enter neither the queue nor the system.

- **System with a finite queue**  
  A queueing system with a limit on the queue length (often folded into the overall capacity parameter $K$).

---

## 2. Arrival and Service Processes

- **Arrival process**  
  A stochastic process describing the arrival times of jobs at the system.

- **Interarrival time**  
  A random variable — the interval between the arrival instants of consecutive jobs.

- **Poisson (simple) process**  
  An arrival process in which:
  - the numbers of arrivals on non-overlapping intervals are independent,
  - the number of arrivals in an interval of length $t$ has a Poisson distribution with parameter $\lambda t$,
  - interarrival times are independent and exponentially distributed with parameter $\lambda$.

- **Arrival rate $\lambda$**  
  The mean number of jobs arriving per unit of time (the parameter of the Poisson process).

- **Service process**  
  A stochastic process describing the service times of jobs at each channel.

- **Service time**  
  A random variable — the time a channel spends serving one job.

- **Service rate $\mu$**  
  The parameter of the exponential service time distribution: the mean number of service completions a channel can perform per unit of time.

- **Total service rate $c\mu$**  
  In a system with $c$ identical channels: the maximum mean throughput (when all channels are busy).

- **Orderly process**  
  A process in which the probability of two or more jobs arriving simultaneously is negligible (usually assumed for real systems).

- **Stationary process**  
  A process whose statistical characteristics are invariant under time shifts (distributions depend only on the interval length, not on its position).

---

## 3. Service Disciplines

- **Service discipline**  
  The rule for choosing the next job from the queue for service.

Main disciplines:

- **FIFO (First In – First Out) / FCFS (First Come – First Served)**  
  The job that arrived earliest is served first (the classic waiting line).

- **LIFO (Last In – First Out) / LCFS (Last Come – First Served)**  
  The most recently arrived job is served first (a stack).

- **SIRO (Service In Random Order)**  
  The next job to be served is picked from the queue at random.

- **Priority service**  
  Jobs are divided into priority classes. Various rules exist:
  - **strict (absolute) priority** — a high-priority job is served before any low-priority one,
  - **relative priority** — priority influences the selection probabilities but does not fully exclude other classes.

- **Preemptive priority**  
  Service of a low-priority job may be interrupted upon the arrival of a higher-priority job.

- **Non-preemptive priority**  
  Service in progress is never interrupted; priorities are taken into account when choosing the next job after the current service completes.

- **Processor Sharing (PS)**  
  All jobs share the server capacity simultaneously (each receiving some fraction of the resource).

---

## 4. System States and Stochastic Processes

- **State of a queueing system**  
  The set of quantities describing the system at the current instant: usually the number of jobs in the system (in service + in the queue).

- **State space**  
  The set of possible values of the number of jobs in the system (e.g., $\{0,1,2,\dots\}$ or $\{0,1,\dots,K\}$).

- **Markov process**  
  A stochastic process whose future depends on the past only through the current state (the memorylessness property).

- **Birth–death process**  
  A special kind of Markov process with a discrete state space, where transitions are possible only between neighboring states:
  - "birth" — an increase in the number of jobs (an arrival),
  - "death" — a decrease in the number of jobs (a service completion).

- **Transition probabilities / transition rates**  
  The probabilities (or rates) of a change of the system state over a small time interval.

- **Stationary (steady-state) regime**  
  A regime in which the distribution of system states does not change over time; the characteristics (mean counts, times, etc.) are stable.

- **System stability**  
  The property of a queueing system whereby the mean number of jobs in the system does not grow without bound.  
  For the simple $M/M/1$ system, the stability condition is $\rho = \dfrac{\lambda}{\mu} < 1$.

---

## 5. Main Quantitative Characteristics

- **Number of jobs in the system $N(t)$**  
  A random variable — the number of jobs in service and in the queue at time $t$.

- **Queue length $Q(t)$**  
  A random variable — the number of jobs in the queue (not being served) at time $t$.

- **Mean number of jobs in the system $L$**  
  The expectation of $N(t)$ in the steady state: $L = \mathbb{E}[N]$.

- **Mean number of jobs in the queue $L_q$**  
  The expectation of the queue length: $L_q = \mathbb{E}[Q]$.

- **Waiting time in the queue $W_q$**  
  A random variable — the time from a job's arrival until the start of its service.

- **Mean waiting time $E[W_q]$**  
  The expectation of the waiting time in the queue.

- **Sojourn time in the system $W$**  
  A random variable — the total time from a job's arrival at the system until its service completes (waiting + service).

- **Mean sojourn time $E[W]$**  
  The expectation of $W$.

- **Channel utilization factor $\rho$**  
  The ratio of the mean arrival rate to the service rate:  
  - for $M/M/1$: $\rho = \dfrac{\lambda}{\mu}$;  
  - for $M/M/c$: $\rho = \dfrac{\lambda}{c\mu}$ (mean load per channel $\lambda/(c\mu)$, total $\lambda/\mu$).

- **Blocking (loss) probability $P_{\text{loss}}$**  
  The probability that a new job is not admitted to the system (e.g., all places are occupied and queueing is not allowed).

- **Waiting probability $P_{\text{wait}}$**  
  The probability that an arriving job does not start service immediately and has to wait.

- **Idle (empty system) probability $P_0$**  
  The probability that there is not a single job in the system.

- **Throughput / departure rate**  
  The mean number of jobs leaving the system per unit of time (in the steady state, usually equal to $\lambda_{\text{eff}}$, the effective arrival rate).

- **Little's Law**  
  A fundamental relationship for stable systems in the steady state:  
  $$L = \lambda_{\text{eff}} \cdot E[W],$$  
  where $L$ is the mean number of jobs in the system, $\lambda_{\text{eff}}$ is the mean arrival (or departure) rate, and $E[W]$ is the mean sojourn time in the system.  
  Similarly for the queue: $$L_q = \lambda_{\text{eff}} \cdot E[W_q].$$

---

## 6. Kendall's Notation and Standard Models

- **Kendall's notation**  
  The standard form for classifying queueing systems:  
  $$A/B/c/K/N/\text{Disc}$$  
  where  
  - $A$ — the interarrival time distribution,  
  - $B$ — the service time distribution,  
  - $c$ — the number of channels,  
  - $K$ — the system capacity (maximum number of jobs in the system, including those in service and in the queue),  
  - $N$ — the source size (maximum possible number of jobs in the source),  
  - $\text{Disc}$ — the service discipline.  
  Most often the shortened form $A/B/c$ is used (the remaining parameters are assumed to be the defaults: $K=\infty$, $N=\infty$, discipline — FIFO).

Main distribution symbols:

- **$M$ (Markovian)** — exponential (for intervals — a Poisson process).
- **$D$ (Deterministic)** — deterministic intervals/times (no variability).
- **$G$ (General)** — a general (arbitrary) distribution.
- **$E_k$ (Erlang)** — the Erlang distribution with parameter $k$ (the sum of $k$ independent exponentials with the same parameter).
- **$H_k$ (Hyperexponential)** — the hyperexponential distribution (a mixture of several exponentials).

Examples of classical models:

- **$M/M/1$**  
  The simplest system: Poisson arrivals, exponential service, one channel, unlimited queue and source, FIFO discipline.

- **$M/M/c$**  
  Poisson arrivals, exponential service, $c$ parallel channels, unlimited queue, FIFO.

- **$M/M/1/K$**  
  Like $M/M/1$, but the system capacity is limited to $K$ (jobs are rejected when $N(t) = K$).

- **$M/M/c/c$ (Erlang B formula)**  
  $c$ channels, capacity $K=c$, no queue; when all channels are busy, new jobs are rejected.

- **$M/M/c/\infty$ with waiting (Erlang C formula)**  
  $c$ channels, unlimited queue; the quantities of interest are the waiting probability and the mean waiting time.

- **$M/G/1$**  
  Poisson arrivals, a single server with a general service time distribution.

- **$G/M/1$**  
  A general arrival process, exponential service, one channel.

- **$GI/G/1$**  
  The most general single-channel model: independent (but not necessarily exponential) interarrival times and an arbitrary service time distribution.

---

## 7. Queueing Networks

- **Queueing network**  
  A system consisting of several nodes (queueing systems) between which jobs can move.

- **Open queueing network**  
  A network into which jobs arrive from the outside and from which they can leave.

- **Closed queueing network**  
  A network with a fixed number of jobs that continually circulate between the nodes (no arrivals from or departures to the outside).

- **Jackson network**  
  A class of open networks with Poisson inputs and exponential service, whose probabilistic structure allows the stationary distribution to be expressed as a product of per-node distributions.

- **Gordon–Newell network**  
  A class of closed queueing networks with exponential service that also admit a product-form (per-node factorized) stationary distribution.

- **Routing**  
  The rules for jobs moving from one network node to another (the transition probability matrix).

---

## 8. Additional Concepts and Properties

- **PASTA (Poisson Arrivals See Time Averages)**  
  A property of systems with Poisson arrivals: the system state "seen" by arriving jobs has the same distribution as the system state at an arbitrary point in time (in the steady state).

- **Loss rate**  
  The mean rate at which jobs are lost (e.g., due to rejections when the system is full).

- **Effective arrival rate $\lambda_{\text{eff}}$**  
  The rate of jobs that actually enter the system (for loss systems, $\lambda_{\text{eff}} < \lambda$).

- **Regularity / variability of the arrival process**  
  A characteristic of the dispersion of interarrival intervals (the coefficient of variation, more detailed process parameters).

- **System with a waiting time limit**  
  A queueing system where a job leaves the queue (is lost or departs) once a certain allowed waiting time is exceeded.

- **System with impatient customers (reneging, balking)**  
  - **Balking** — refusing to join the queue if it is too long.  
  - **Reneging** — leaving the queue if the wait drags on.
