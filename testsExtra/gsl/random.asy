import TestLib;
import gsl;

StartTest("random number generators");
rng_init();
assert(rng_min() == 0);
assert(rng_max() == 4294967295);
assert(rng_get() == 4293858116);
rng_init("taus2");
assert(rng_min() == 0);
assert(rng_max() == 4294967295);
assert(rng_get() == 802792108);
rng_init("gfsr4");
assert(rng_min() == 0);
assert(rng_max() == 4294967295);
assert(rng_get() == 2901276280);
string[] list = rng_list();
for(string name: list)
{
  rng_init(name);
  rng_min();
  rng_max();
  rng_get();
}
assert(list.length >= 62);
rng_init();
rng_set(1);
assert(rng_get() == 1791095845);
rng_set(1);
assert(rng_get() == 1791095845);
EndTest();

StartTest("Bernoulli distribution");
assert(close(pdf_bernoulli(0,0.3), 0.7));
assert(close(pdf_bernoulli(1,0.3), 0.3));
//rng_init();
//assert(rng_bernoulli(0.3) == 0);
EndTest();

StartTest("beta distribution");
assert(close(cdf_beta_P(0.3,5,5), 0.09880866));
assert(close(cdf_beta_Q(0.3,5,5) + cdf_beta_P(0.3,5,5), 1));
assert(close(cdf_beta_Pinv(cdf_beta_P(0.3,5,5),5,5), 0.3));
assert(close(cdf_beta_Qinv(cdf_beta_Q(0.3,5,5),5,5), 0.3));
assert(close(pdf_beta(0.3,5,5), 1.2252303));
//rng_init();
//assert(close(rng_beta(5,5), 0.533021338130471));
EndTest();

StartTest("binomial distribution");
assert(close(cdf_binomial_P(5,0.3,10), 0.9526510126));
assert(close(cdf_binomial_P(5,0.3,10) + cdf_binomial_Q(5,0.3,10), 1));
assert(close(pdf_binomial(5,0.3,10), 0.1029193452));
//rng_init();
//assert(rng_binomial(0.3,10) == 8);
EndTest();

StartTest("bivariate Gaussian distribution");
assert(close(pdf_bivariate_gaussian((1,1),(0,2),(4,6),0.5),
             0.00675758392382108));
//rng_init();
//pair z = (-0.260388644979556,2.50057001628669);
//pair r = rng_bivariate_gaussian((0,2),(4,6),0.5);
//assert(close(length(r - z), 0));
EndTest();

StartTest("cauchy distribution");
assert(close(cdf_cauchy_P(1,3), 0.602416382349567));
assert(close(cdf_cauchy_P(1,3) + cdf_cauchy_Q(1,3), 1));
assert(close(cdf_cauchy_Pinv(cdf_cauchy_P(1,3),3), 1));
assert(close(cdf_cauchy_Qinv(cdf_cauchy_Q(1,3),3), 1));
assert(close(pdf_cauchy(1,3), 0.0954929658551372));
//rng_init();
//assert(close(rng_cauchy(3), -0.0024339597467863));
EndTest();

StartTest("chi-squared distribution");
assert(close(cdf_chisq_P(4,6), 0.323323583816936));
assert(close(cdf_chisq_P(4,6) + cdf_chisq_Q(4, 6), 1));
assert(close(cdf_chisq_Pinv(cdf_chisq_P(4,6),6), 4));
assert(close(cdf_chisq_Qinv(cdf_chisq_Q(4,6),6), 4));
assert(close(pdf_chisq(4,6), 0.135335283236613));
//rng_init();
//assert(close(rng_chisq(6), 8.24171826270279));
EndTest();

StartTest("Dirichlet distribution");
real[] alpha = {1,2,3,4};
real[] theta = {0.1,0.2,0.3,0.4};
assert(close(pdf_dirichlet(alpha,theta), 34.83648));
//rng_init();
//real[] z = {0.124480735441317,
//            0.191823537067349,
//            0.460543885448264,
//            0.22315184204307};
//real[] r = rng_dirichlet(alpha);
//assert(close(norm(r - z), 0));
EndTest();

StartTest("exponential distribution");
assert(close(cdf_exponential_P(2,3), 0.486582880967408));
assert(close(cdf_exponential_P(2,3) + cdf_exponential_Q(2,3), 1));
assert(close(cdf_exponential_Pinv(cdf_exponential_P(2,3),3), 2));
assert(close(cdf_exponential_Qinv(cdf_exponential_Q(2,3),3), 2));
assert(close(pdf_exponential(2,3), 0.171139039677531));
//rng_init();
//assert(close(rng_exponential(3), 24.7847346491112));
EndTest();

StartTest("exponential power distribution");
assert(close(cdf_exppow_P(2,3,2), 0.82711070692442));
assert(close(cdf_exppow_P(2,3,2) + cdf_exppow_Q(2,3,2), 1));
assert(close(pdf_exppow(2,3,2), 0.120582432109095));
//rng_init();
//assert(close(rng_exppow(3,2), 0.284084267783339));
EndTest();

StartTest("F-distribution");
assert(close(cdf_fdist_P(1,5,4), 0.485657196759213));
assert(close(cdf_fdist_P(1,5,4) + cdf_fdist_Q(1,5,4), 1));
//rng_init();
//assert(close(rng_fdist(5,4), 1.20570928490019));
EndTest();

StartTest("flat (uniform) distribution");
assert(close(cdf_flat_P(2,0,5), 0.4));
assert(close(cdf_flat_P(2,0,5) + cdf_flat_Q(2,0,5), 1));
assert(close(cdf_flat_Pinv(cdf_flat_P(2,0,5),0,5), 2));
assert(close(cdf_flat_Qinv(cdf_flat_Q(2,0,5),0,5), 2));
assert(close(pdf_flat(2,0,5), 0.2));
//rng_init();
//assert(close(rng_flat(0,5), 4.99870874453336));
EndTest();

StartTest("Gamma-distribution");
assert(close(cdf_gamma_P(6,5,1), 0.71494349968337));
assert(close(cdf_gamma_P(6,5,1) + cdf_gamma_Q(6,5,1), 1));
assert(close(cdf_gamma_Pinv(cdf_gamma_P(6,5,1),5,1), 6));
assert(close(cdf_gamma_Qinv(cdf_gamma_Q(6,5,1),5,1), 6));
assert(close(pdf_gamma(6,5,1), 0.133852617539983));
//rng_init();
//assert(close(rng_gamma(5,1), 6.52166444209317));
//assert(close(rng_gamma(5,1,"mt"), 5.71361391461836));
//assert(close(rng_gamma(5,1,"knuth"), 1.53054227085541));
EndTest();

StartTest("Gaussian distribution");
assert(close(cdf_gaussian_P(1,0,1), 0.841344746068543));
assert(close(cdf_gaussian_P(1,0,1) + cdf_gaussian_Q(1,0,1), 1));
assert(close(cdf_gaussian_Pinv(cdf_gaussian_P(1,0,1),0,1), 1));
assert(close(cdf_gaussian_Qinv(cdf_gaussian_Q(1,0,1),0,1), 1));
assert(close(pdf_gaussian(1,0,1), 0.241970724519143));
//rng_init();
//assert(close(rng_gaussian(0,1), 0.133918608118676));
//assert(close(rng_gaussian(1,2,"ziggurat"), 1.90467233084303));
//assert(close(rng_gaussian(1,2,"ratio"), 4.04779517509342));
//assert(close(rng_gaussian(1,2,"polar"), 1.54245166575101));
EndTest();

StartTest("Gaussian tail distribution");
assert(close(pdf_gaussian_tail(2,1,1), 0.34030367841782));
//rng_init();
//assert(close(rng_gaussian_tail(1,1), 1.0528474462339));
EndTest();

StartTest("geometric distribution");
assert(close(cdf_geometric_P(6,0.1), 0.468559));
assert(close(cdf_geometric_P(6,0.1) + cdf_geometric_Q(6,0.1), 1));
assert(close(pdf_geometric(6,0.1), 0.059049));
//rng_init();
//assert(rng_geometric(0.1) == 1);
EndTest();

StartTest("Gumbel1 distribution");
assert(close(cdf_gumbel1_P(1,3,8), 0.671462877871127));
assert(close(cdf_gumbel1_P(1,3,8) + cdf_gumbel1_Q(1,3,8), 1));
assert(close(cdf_gumbel1_Pinv(cdf_gumbel1_P(1,3,8),3,8), 1));
assert(close(cdf_gumbel1_Qinv(cdf_gumbel1_Q(1,3,8),3,8), 1));
assert(close(pdf_gumbel1(1,3,8), 0.80232403696926));
//rng_init();
//assert(close(rng_gumbel1(3,8), 3.44696353953564));
EndTest();

StartTest("Gumbel2 distribution");
assert(close(cdf_gumbel2_P(2,2,3), 0.472366552741015));
assert(close(cdf_gumbel2_P(2,2,3) + cdf_gumbel2_Q(2,2,3), 1));
assert(close(cdf_gumbel2_Pinv(cdf_gumbel2_P(2,2,3),2,3), 2));
assert(close(cdf_gumbel2_Qinv(cdf_gumbel2_Q(2,2,3),2,3), 2));
assert(close(pdf_gumbel2(2,2,3), 0.354274914555761));
//rng_init();
//assert(close(rng_gumbel2(2,3), 107.773379309453));
EndTest();

StartTest("hypergeometric distribution");
assert(close(cdf_hypergeometric_P(4,10,10,8), 0.675041676589664));
assert(close(cdf_hypergeometric_P(4,10,10,8) +
             cdf_hypergeometric_Q(4,10,10,8), 1));
assert(close(pdf_hypergeometric(4,10,10,8), 0.350083353179329));
//rng_init();
//assert(rng_hypergeometric(10,10,8) == 3);
EndTest();

StartTest("Laplace distribution");
assert(close(cdf_laplace_P(1,2), 0.696734670143683));
assert(close(cdf_laplace_P(1,2) + cdf_laplace_Q(1,2), 1));
assert(close(cdf_laplace_Pinv(cdf_laplace_P(1,2),2), 1));
assert(close(cdf_laplace_Qinv(cdf_laplace_Q(1,2),2), 1));
assert(close(pdf_laplace(1,2), 0.151632664928158));
//rng_init();
//assert(close(rng_laplace(2), 0.00103327123971616));
EndTest();

StartTest("Landau distribution");
assert(close(pdf_landau(1), 0.145206637130862));
//rng_init();
//assert(close(rng_landau(), 3880.0374262546));
EndTest();

//StartTest("Levy stable distribution");
//rng_init();
//assert(close(rng_levy(1,1,0), 1232.55941432972));
//assert(close(rng_levy(1,1,1), -0.13781830409645));
//EndTest();

StartTest("logistic distribution");
assert(close(cdf_logistic_P(1,2), 0.622459331201855));
assert(close(cdf_logistic_P(1,2) + cdf_logistic_Q(1,2), 1));
assert(close(cdf_logistic_Pinv(cdf_logistic_P(1,2),2), 1));
assert(close(cdf_logistic_Qinv(cdf_logistic_Q(1,2),2), 1));
assert(close(pdf_logistic(1,2), 0.117501856100797));
//rng_init();
//assert(close(rng_logistic(2), 16.522639863849));
EndTest();

StartTest("lognormal distribution");
assert(close(cdf_lognormal_P(6,2,1), 0.417520581602749));
assert(close(cdf_lognormal_P(6,2,1) + cdf_lognormal_Q(6,2,1), 1));
assert(close(cdf_lognormal_Pinv(cdf_lognormal_P(6,2,1),2,1), 6));
assert(close(cdf_lognormal_Qinv(cdf_lognormal_Q(6,2,1),2,1), 6));
assert(close(pdf_lognormal(6,2,1), 0.0650642483079156));
//rng_init();
//assert(close(rng_lognormal(2,1), 6.92337133931968));
EndTest();

StartTest("multinomial distribution");
real[] p = {0.1,0.2,0.3,0.4};
int[] n = {1,2,3,4};
assert(close(pdf_multinomial(p,n), 0.03483648));
//rng_init();
//int[] r = {5, 0, 1, 4};
//assert(all(rng_multinomial(10,p) == r));
EndTest();

StartTest("negative binomial distribution");
assert(close(cdf_negative_binomial_P(6,0.5,10), 0.227249145507813));
assert(close(cdf_negative_binomial_P(6,0.5,10) +
             cdf_negative_binomial_Q(6,0.5,10), 1));
assert(close(pdf_negative_binomial(6,0.5,10), 0.076370239257812));
//rng_init();
//assert(rng_negative_binomial(0.5,10) == 15);
EndTest();

StartTest("Pareto distribution");
assert(close(cdf_pareto_P(4,2,2), 0.75));
assert(close(cdf_pareto_P(4,2,2) + cdf_pareto_Q(4,2,2), 1));
assert(close(cdf_pareto_Pinv(cdf_pareto_P(4,2,2),2,2), 4));
assert(close(cdf_pareto_Qinv(cdf_pareto_Q(4,2,2),2,2), 4));
assert(close(pdf_pareto(4,2,2), 0.125));
//rng_init();
//assert(close(rng_pareto(2,2), 2.00025830112432));
EndTest();

StartTest("Poisson distribution");
assert(close(cdf_poisson_P(5,6), 0.445679641364611));
assert(close(cdf_poisson_P(5,6) + cdf_poisson_Q(5,6), 1));
assert(close(pdf_poisson(5,6), 0.16062314104798));
//rng_init();
//assert(rng_poisson(6) == 8);
EndTest();

StartTest("Rayleigh distribution");
assert(close(cdf_rayleigh_P(3,2), 0.67534753264165));
assert(close(cdf_rayleigh_P(3,2) + cdf_rayleigh_Q(3,2), 1));
assert(close(cdf_rayleigh_Pinv(cdf_rayleigh_P(3,2),2), 3));
assert(close(cdf_rayleigh_Qinv(cdf_rayleigh_Q(3,2),2), 3));
assert(close(pdf_rayleigh(3,2), 0.243489350518762));
//rng_init();
//assert(close(rng_rayleigh(2), 0.0454563039310455));
EndTest();

StartTest("Rayleigh tail distribution");
assert(close(pdf_rayleigh_tail(5,4,1), 0.0555449826912115));
//rng_init();
//assert(close(rng_rayleigh_tail(4,1), 4.0000645705903));
EndTest();

//StartTest("spherical distributions");
//rng_init();
//pair z = (-0.617745613497854,-0.786377998804748);
//pair r = rng_dir2d();
//assert(close(length(r - z), 0));
//pair z = (0.993748310886084,0.111643605329884);
//pair r = rng_dir2d("neumann");
//assert(close(length(r - z), 0));
//pair z = (0.964519203132591,-0.264012701945327);
//pair r = rng_dir2d("trig");
//assert(close(length(r - z), 0));
//triple z = (0.849028025629996,0.139162687752509,-0.509691237939527);
//triple r = rng_dir3d();
//assert(close(length(r - z), 0));
//real[] z = {0.420990368676528, 
//            -0.626782975357296,
//            0.0441585572224004,
//            -0.0458388920727644,
//            -0.652578753164271};
//real[] r = rng_dir(5);
//assert(close(norm(r - z), 0));
//EndTest();

StartTest("t-distribution");
assert(close(cdf_tdist_P(0.6,2), 0.695283366471236));
assert(close(cdf_tdist_P(0.6,2) + cdf_tdist_Q(0.6,2), 1));
assert(close(cdf_tdist_Pinv(cdf_tdist_P(0.6,2),2), 0.6));
assert(close(cdf_tdist_Qinv(cdf_tdist_Q(0.6,2),2), 0.6));
assert(close(pdf_tdist(0.6,2), 0.275823963942424));
//rng_init();
//assert(close(rng_tdist(2), 0.127201714006725));
EndTest();

StartTest("Weibull distribution");
assert(close(cdf_weibull_P(1,2,2), 0.221199216928595));
assert(close(cdf_weibull_P(1,2,2) + cdf_weibull_Q(1,2,2), 1));
assert(close(cdf_weibull_Pinv(cdf_weibull_P(1,2,2),2,2), 1));
assert(close(cdf_weibull_Qinv(cdf_weibull_Q(1,2,2),2,2), 1));
assert(close(pdf_weibull(1,2,2), 0.389400391535702));
//rng_init();
//assert(close(rng_weibull(2,2), 0.032142460757319));
EndTest();
