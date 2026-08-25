# Q2 V3 revised power, precision, and cost plan

Status: `DESIGN ONLY — NO EXECUTION AUTHORIZED`

## Precision logic

The primary evidence is family-balanced and dyad-dependent. Naive independent
pair power is forbidden. Before freeze, simulation must jointly resample:

- item clusters, moving every condition and both rollouts;
- complete controller directions, moving both shells together;
- complete conceptual families;
- shell-specific dyad matrices.

Simulations must span Regime 0 through Regime 4, vary itemwise error prevalence,
family heterogeneity, radial leakage, angular effect rho, and M3 numerical
noise. They must estimate:

- family-balanced rho precision;
- max-statistic QAP behavior across M0–M3;
- probability of 4/5 positive families;
- paired metric-contrast precision;
- false attribution of radial structure to angular geometry.

Q2 V2 is a development input, not confirmatory evidence. Its current residual
rho near 0.054 is an explicit low-effect scenario. The revised design does not
claim that N=200 can rescue it; the matched-shell bank is the identifying
change.

## Draft scale

| Phase | Rows/operations |
|---|---:|
| same-domain baseline instrument qualification | 40 items x 2 = 80 |
| new source qualification | approximately 400–600 trajectories |
| physical-shell calibration and safety | approximately 600–1,000 trajectories |
| M3 engineering qualification | no scientific rows |
| M0/M1/M2/M3 pre-outcome captures | teacher-forced operations, counted separately |
| primary panel | 200 x 25 x 2 = 10,000 trajectories |
| optional transfer panel | draft 120 x 25 x 2 = 6,000 trajectories |

Exact schedules remain to be frozen. The optional transfer panel requires a
separate authorization after the primary result and must not be bundled into an
unattended transition.

## Runtime and cost

### M3 engineering qualification

- expected: 0.5–1.5 A40 GPU-h, US$0.25–0.75;
- conservative: 4 GPU-h, US$2 planning ceiling;
- no scientific trajectories.

### Q2 V3 primary program

Using Q2 V2's observed throughput only as one development anchor:

- expected generation/capture runtime: 7–12 A40 GPU-h;
- conservative runtime: 24 GPU-h;
- expected total cost: US$4–7;
- conservative planning envelope: US$12;
- proposed hard ceiling before future freeze: US$15;
- wallet gate: verified balance at least US$18 and projected cost with a 50%
  tail margin within the hard ceiling.

The estimate includes source and shell calibration plus the 10,000-row panel.
It does not authorize spending.

### Optional transfer panel

- expected: 4–7 A40 GPU-h;
- expected cost: US$2–4;
- conservative envelope: US$8;
- separately frozen and authorized only after primary closeout.

## Stop rules before spending

Stop before the primary panel if:

- M3 is required by the future design but does not qualify;
- no fresh same-domain instrument is available;
- the radial/angular identifiability gate fails;
- controller safety/manipulation fails;
- projected cost exceeds the future authorized ceiling.

Do not reduce families, shells, items, conditions, rollouts, or token cap merely
to fit a wallet balance.
