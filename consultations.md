# 27.03.2025

Questions and answers:

1. Keep all the function definitions in one separate section or in the relevant sections as it is right now. - In one section and add better docstrings, output must be there, could also add params
2. What is meant by *iteration* here: Likelihood function values and coefficient values depending on *iteration* - *Iteration* of the middle loop or the lambda parameter? - generally the inner (middle) loop, but it depends on us if we want to show outer or inner
3. How long should the methodology section be in the report. Should we explain what Log Reg is from ground up or just discuss the performance improvements? - short, just the optimizations, but the section in notebook can stay
4. Is it ok if we merge the comparison of LogRegCCD with Log Reg no reg and LogRegCCD with Log Reg L1 - yes, in general we should produce plots that show as much as possible
5. How to place legend on multiplots, only 1? - yes, only on the first graph is fine, but change the colors of the lines
6. Is creating a specific class for each real dataset ok?\ - yes as long as there is instruction of how to use a custom prepared dataset (how to use Dataset class basically)
7. How many n,p,d,g params to check? - what we have is bsaically fine, we could add more if we wish to. We should examine the case when n is low and d is high, and then check the effect of g
9. Can we compare our solution to glmnet? - yes
10. Can we change the margins in the report - yes, even the font, but in a reasonable way
11. If we need more consultations, we can contact prof on teams on Sunday, Monday, Tuesday

Discussion of the correctness:
1. Generate a dataset with some redundant varaibles and check whether they get set to 0
2. Compare with some ready implementation of ccg, compare the lambda values, coefficients paths
3. Set beta start, as a result of log reg without l1, then less itration wiht small lambdas