# Q2 V4 K/N decision

Recommendation:

\[
\boxed{K=32,\quad N=300}
\]

This produces 32 independently sampled intervention directions, two deployed
amplitude shells, baseline, and two independent rollouts:

\[
(2K+1)N2=65\times300\times2=39{,}000
\]

semantic trajectories.

K=16/20/24 fail to provide stable >=80% power around the scientifically
meaningful rho=0.25 target. K=32/N=200 reaches about 84% omnibus power, but N=300
raises it to about 94%, materially improves radial power, and reduces planning
rho width by about 20%. The choice is not significance-optimized and uses no
semantic outcome.

The final bank itself is not generated in this sprint. It depends on a
prospectively qualified Spark-native source basis and future lock commit.
