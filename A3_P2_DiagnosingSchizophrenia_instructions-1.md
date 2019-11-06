Assignment 3 - Part 2 - Diagnosing schizophrenia from voice
-----------------------------------------------------------

In the previous part of the assignment you generated a bunch of
“features”, that is, of quantitative descriptors of voice in
schizophrenia. We then looked at whether we could replicate results from
the previous literature. We now want to know whether we can
automatically diagnose schizophrenia from voice only, that is, relying
on the set of features you produced last time, we will try to produce an
automated classifier. Again, remember that the dataset containst 7
studies and 3 languages. Feel free to only include Danish (Study 1-4) if
you feel that adds too much complexity.

Issues to be discussed your report: - Should you run the analysis on all
languages/studies at the same time? - Choose your best acoustic feature
from part 1. How well can you diagnose schizophrenia just using it? -
Identify the best combination of acoustic features to diagnose
schizophrenia using logistic regression. - Discuss the “classification”
process: which methods are you using? Which confounds should you be
aware of? What are the strength and limitation of the analysis? - Bonus
question: Logistic regression is only one of many classification
algorithms. Try using others and compare performance. Some examples:
Discriminant Function, Random Forest, Support Vector Machine, etc. The
package caret provides them. - Bonus Bonus question: It is possible
combine the output of multiple classification models to improve
classification accuracy. For inspiration see,
<a href="https://machinelearningmastery.com/machine-learning-ensembles-with-r/" class="uri">https://machinelearningmastery.com/machine-learning-ensembles-with-r/</a>
The interested reader might also want to look up ‘The BigChaos Solution
to the Netflix Grand Prize’

Learning objectives
-------------------

-   Learn the basics of classification in a machine learning framework
-   Design, fit and report logistic regressions
-   Apply feature selection techniques

### Let’s start

We first want to build a logistic regression to see whether you can
diagnose schizophrenia from your best acoustic feature. Let’s use the
full dataset and calculate the different performance measures (accuracy,
sensitivity, specificity, PPV, NPV, ROC curve). You need to think
carefully as to how we should (or not) use study and subject ID.

Then cross-validate the logistic regression and re-calculate performance
on the testing folds. N.B. The cross-validation functions you already
have should be tweaked: you need to calculate these new performance
measures. Alternatively, the groupdata2 and cvms package created by
Ludvig are an easy solution.

N.B. the predict() function generates log odds (the full scale between
minus and plus infinity). Log odds &gt; 0 indicates a choice of 1, below
a choice of 0. N.N.B. you need to decide whether calculate performance
on each single test fold or save all the prediction for test folds in
one dataset, so to calculate overall performance. N.N.N.B. Now you have
two levels of structure: subject and study. Should this impact your
cross-validation? N.N.N.N.B. A more advanced solution could rely on the
tidymodels set of packages (warning: Time-consuming to learn as the
documentation is sparse, but totally worth it)

    # load
    library(tidyverse,pacman)

    ## ── Attaching packages ────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✔ ggplot2 3.2.1     ✔ purrr   0.3.2
    ## ✔ tibble  2.1.3     ✔ dplyr   0.8.3
    ## ✔ tidyr   1.0.0     ✔ stringr 1.4.0
    ## ✔ readr   1.3.1     ✔ forcats 0.4.0

    ## ── Conflicts ───────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## ✖ dplyr::filter() masks stats::filter()
    ## ✖ dplyr::lag()    masks stats::lag()

    pacman::p_load(purrr,janitor, lme4, parsnip, groupdata2, tidymodels, caret, e1071, pROC, cvms, kernlab, randomForest, xgboost)

    # load csv
    feature_df <- read.csv("acoustic_features_by_language.csv")

1.  Find best feature from A3P1 (by AIC and BIC on training data)

<!-- -->

1.  by creating logistic regressions for each feature (all 4)
2.  Think carefully as to how we should (or not) use study and subject
    ID
3.  Then run ANOVA

<!-- -->

    ##Making logistic regression models for choosing best feature by predicting diagnosis from acoustic feature##
    #pitch variablity
    model1 <- glmer(Diagnosis ~ PitchVariability + (1|uID), data = feature_df, family = "binomial")

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl =
    ## control$checkConv, : Model failed to converge with max|grad| = 0.124879
    ## (tol = 0.001, component 1)

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    summary(model1)

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: binomial  ( logit )
    ## Formula: Diagnosis ~ PitchVariability + (1 | uID)
    ##    Data: feature_df
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##    516.9    535.6   -255.4    510.9     3815 
    ## 
    ## Scaled residuals: 
    ##       Min        1Q    Median        3Q       Max 
    ## -0.001028 -0.001001  0.010163  0.019310  0.033259 
    ## 
    ## Random effects:
    ##  Groups Name        Variance Std.Dev.
    ##  uID    (Intercept) 5212     72.19   
    ## Number of obs: 3818, groups:  uID, 315
    ## 
    ## Fixed effects:
    ##                    Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept)      -1.377e+01  5.663e-04 -24309.6   <2e-16 ***
    ## PitchVariability -9.075e-02  5.662e-04   -160.3   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## PitchVrblty 0.000 
    ## convergence code: 0
    ## Model failed to converge with max|grad| = 0.124879 (tol = 0.001, component 1)
    ## Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    #phonatation time
    model2 <- glmer(Diagnosis ~ PhonatationTime + (1|uID), data = feature_df, family = "binomial")

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model failed to converge with max|grad| = 0.125932 (tol = 0.001, component 1)

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    summary(model2)

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: binomial  ( logit )
    ## Formula: Diagnosis ~ PhonatationTime + (1 | uID)
    ##    Data: feature_df
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##    516.9    535.6   -255.4    510.9     3815 
    ## 
    ## Scaled residuals: 
    ##       Min        1Q    Median        3Q       Max 
    ## -0.001096 -0.001007  0.009756  0.019002  0.037886 
    ## 
    ## Random effects:
    ##  Groups Name        Variance Std.Dev.
    ##  uID    (Intercept) 5168     71.89   
    ## Number of obs: 3818, groups:  uID, 315
    ## 
    ## Fixed effects:
    ##                   Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)     -1.373e+01  5.616e-04  -24451   <2e-16 ***
    ## PhonatationTime -1.308e-01  5.616e-04    -233   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## PhonatatnTm 0.000 
    ## convergence code: 0
    ## Model failed to converge with max|grad| = 0.125932 (tol = 0.001, component 1)
    ## Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    #speech rate
    model3 <- glmer(Diagnosis ~ Speechrate + (1|uID), data = feature_df, family = "binomial")

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model failed to converge with max|grad| = 0.124742 (tol = 0.001, component 1)

    ## Warning in checkConv(attr(opt, "derivs"), opt$par, ctrl = control$checkConv, : Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    summary(model3)

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: binomial  ( logit )
    ## Formula: Diagnosis ~ Speechrate + (1 | uID)
    ##    Data: feature_df
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##    516.9    535.6   -255.4    510.9     3815 
    ## 
    ## Scaled residuals: 
    ##       Min        1Q    Median        3Q       Max 
    ## -0.001188 -0.000980  0.009897  0.018861  0.031707 
    ## 
    ## Random effects:
    ##  Groups Name        Variance Std.Dev.
    ##  uID    (Intercept) 5185     72      
    ## Number of obs: 3818, groups:  uID, 315
    ## 
    ## Fixed effects:
    ##               Estimate Std. Error  z value Pr(>|z|)    
    ## (Intercept) -1.376e+01  5.669e-04 -24275.5   <2e-16 ***
    ## Speechrate  -1.195e-01  5.668e-04   -210.8   <2e-16 ***
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##            (Intr)
    ## Speechrate 0.000 
    ## convergence code: 0
    ## Model failed to converge with max|grad| = 0.124742 (tol = 0.001, component 1)
    ## Model is nearly unidentifiable: very large eigenvalue
    ##  - Rescale variables?

    #pause duration
    model4 <- glmer(Diagnosis ~ PauseDuration + (1|uID), data = feature_df, family = "binomial")
    summary(model4)

    ## Generalized linear mixed model fit by maximum likelihood (Laplace
    ##   Approximation) [glmerMod]
    ##  Family: binomial  ( logit )
    ## Formula: Diagnosis ~ PauseDuration + (1 | uID)
    ##    Data: feature_df
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##    517.0    535.7   -255.5    511.0     3815 
    ## 
    ## Scaled residuals: 
    ##       Min        1Q    Median        3Q       Max 
    ## -0.001052 -0.001033  0.010554  0.019292  0.032548 
    ## 
    ## Random effects:
    ##  Groups Name        Variance Std.Dev.
    ##  uID    (Intercept) 4856     69.68   
    ## Number of obs: 3818, groups:  uID, 315
    ## 
    ## Fixed effects:
    ##                 Estimate Std. Error z value Pr(>|z|)    
    ## (Intercept)   -13.696705   0.806600 -16.981   <2e-16 ***
    ## PauseDuration   0.005871   0.388845   0.015    0.988    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr)
    ## PauseDuratn -0.006

    #comparing models
    anova(model1, model2, model3, model4)

    ## Data: feature_df
    ## Models:
    ## model1: Diagnosis ~ PitchVariability + (1 | uID)
    ## model2: Diagnosis ~ PhonatationTime + (1 | uID)
    ## model3: Diagnosis ~ Speechrate + (1 | uID)
    ## model4: Diagnosis ~ PauseDuration + (1 | uID)
    ##        Df    AIC    BIC  logLik deviance  Chisq Chi Df Pr(>Chisq)    
    ## model1  3 516.88 535.63 -255.44   510.88                             
    ## model2  3 516.85 535.59 -255.43   510.85 0.0332      0     <2e-16 ***
    ## model3  3 516.85 535.59 -255.43   510.85 0.0000      0          1    
    ## model4  3 516.97 535.71 -255.48   510.97 0.0000      0          1    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

1.  assess accuracy, sensitivity, PPV, NPV and ROC curves, when
    predicting on all the data the model has been trained on

<!-- -->

    #add 1000 to uID
    feature_df$uID <- feature_df$uID + 1000

    #create a cunfusion matrix of best single features to assess accuracy
    predicted_values <- predict(model2, feature_df, allow.new.levels = T)
    pred_df <- tibble(predictions = predicted_values, actual = feature_df$Diagnosis) 
    pred_df$predictions = ifelse(pred_df$predictions > 0.0, "0", "1") 
    pred_df$predictions <- as_factor(pred_df$predictions)
    pred_df$actual <- as_factor(pred_df$actual)#make sure it is a factor
    caret::confusionMatrix(pred_df$predictions, pred_df$actual, positive ="0")

    ## Warning in confusionMatrix.default(pred_df$predictions, pred_df$actual, :
    ## Levels are not in the same order for reference and data. Refactoring data
    ## to match.

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0    0    0
    ##          1 1789 2029
    ##                                           
    ##                Accuracy : 0.5314          
    ##                  95% CI : (0.5155, 0.5474)
    ##     No Information Rate : 0.5314          
    ##     P-Value [Acc > NIR] : 0.5066          
    ##                                           
    ##                   Kappa : 0               
    ##                                           
    ##  Mcnemar's Test P-Value : <2e-16          
    ##                                           
    ##             Sensitivity : 0.0000          
    ##             Specificity : 1.0000          
    ##          Pos Pred Value :    NaN          
    ##          Neg Pred Value : 0.5314          
    ##              Prevalence : 0.4686          
    ##          Detection Rate : 0.0000          
    ##    Detection Prevalence : 0.0000          
    ##       Balanced Accuracy : 0.5000          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

    #making a ROC curve
    #plot(roc(pred_df$predictions,feature_df$Diagnosis)) # doesn't function since there is only one level in the prediction column

1.  partition (groupdata2)

<!-- -->

1.  remember to take ID into account, gender and Diagnosis should be
    equally distributed across folds
2.  find out whether we should take study into account also

<!-- -->

    # Partitioning the dataframe
    featurefold <- groupdata2::fold(feature_df, k = 5, id_col = "uID", cat_col = c("Diagnosis", "Gender", "Study")) %>% arrange(.folds)

1.  Then cross-validate the logistic regression and re-calculate
    performance on the testing folds (cvms)

<!-- -->

    CV1 <- cvms::cross_validate(featurefold, "Diagnosis ~ PhonatationTime + (1|uID)", 
                          fold_cols = '.folds',
                          family = "binomial",
                          control = glmerControl(
                            optimizer = "nloptwrap",
                            calc.derivs = F,
                            optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8, maxfun = 1000)),
                          REML = FALSE,
                          rm_nc = FALSE
                          )
    head(CV1)

    ## # A tibble: 1 x 29
    ##   `Balanced Accur…    F1 Sensitivity Specificity `Pos Pred Value`
    ##              <dbl> <dbl>       <dbl>       <dbl>            <dbl>
    ## 1            0.518 0.374       0.279       0.756            0.565
    ## # … with 24 more variables: `Neg Pred Value` <dbl>, AUC <dbl>, `Lower
    ## #   CI` <dbl>, `Upper CI` <dbl>, Kappa <dbl>, MCC <dbl>, `Detection
    ## #   Rate` <dbl>, `Detection Prevalence` <dbl>, Prevalence <dbl>,
    ## #   Predictions <list>, ROC <list>, `Confusion Matrix` <list>,
    ## #   Coefficients <list>, Folds <int>, `Fold Columns` <int>, `Convergence
    ## #   Warnings` <int>, `Singular Fit Messages` <int>, `Other
    ## #   Warnings` <int>, `Warnings and Messages` <list>, Family <chr>,
    ## #   Link <chr>, Dependent <chr>, Fixed <chr>, Random <chr>

    cv_plot(CV1, type ='ROC')

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-5-1.png)

1.  The best combination of features?

<!-- -->

1.  test multiple models, and find accuracy, sensitivity, specificity
    for all models
2.  discuss above parameters
3.  choose best model(s)

<!-- -->

    # List of models
    model_list <- c(
      "Diagnosis ~ PitchVariability + (1|uID)",
      "Diagnosis ~ Speechrate + (1|uID)",
      "Diagnosis ~ PhonatationTime + (1|uID)",
      "Diagnosis ~ PauseDuration + (1|uID)",
      "Diagnosis ~ PhonatationTime + PitchVariability + (1|uID)",
      "Diagnosis ~ PhonatationTime + PitchVariability + Speechrate + (1|uID)",
      "Diagnosis ~ PhonatationTime + PitchVariability + Speechrate + PauseDuration + (1|uID)",
      "Diagnosis ~ PitchVariability + Speechrate + (1|uID)",
      "Diagnosis ~ PitchVariability + Speechrate + PauseDuration + (1|uID)",
      "Diagnosis ~ Speechrate + PauseDuration + (1|uID)"
      )

    model_list2 <- c(

      "Diagnosis ~ PhonatationTime * PitchVariability + (1|uID)",
      "Diagnosis ~ PhonatationTime * PitchVariability * PauseDuration + (1|uID)",
      "Diagnosis ~ PhonatationTime * PitchVariability * Speechrate * PauseDuration + (1|uID)",
      "Diagnosis ~ PitchVariability * Speechrate + (1|uID)",
      "Diagnosis ~ PitchVariability * Speechrate * PauseDuration + (1|uID)",
      "Diagnosis ~ Speechrate * PauseDuration + (1|uID)"
      )

    model_list3 <- c(
      "Diagnosis ~ PhonatationTime + PitchVariability + (1|uID)",
      "Diagnosis ~ PhonatationTime + Speechrate + (1|uID)",
      "Diagnosis ~ PhonatationTime + (1|uID)"
      )

    # Cross validate model lists
    CVlist1 <- cvms::cross_validate(featurefold, model_list,
                          family = "binomial",
                          control = glmerControl(
                            optimizer = "nloptwrap",
                             calc.derivs = F,
                            optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8, maxfun = 1000)),
                          REML = FALSE,
                          rm_nc = FALSE
                          )
    CVlist1[,1:4] # model 7 ("Diagnosis ~ PhonatationTime + PitchVariability + Speechrate + PauseDuration + (1|uID)") had the highest accuracy of 0.532 and spens. and spec. somewhat balanced (0.484, 0.580)

    ## # A tibble: 10 x 4
    ##    `Balanced Accuracy`     F1 Sensitivity Specificity
    ##                  <dbl>  <dbl>       <dbl>       <dbl>
    ##  1               0.492 0.283       0.195        0.790
    ##  2               0.516 0.309       0.212        0.820
    ##  3               0.518 0.374       0.279        0.756
    ##  4               0.492 0.0906      0.0503       0.933
    ##  5               0.496 0.446       0.387        0.604
    ##  6               0.512 0.492       0.449        0.575
    ##  7               0.524 0.508       0.466        0.582
    ##  8               0.519 0.387       0.294        0.743
    ##  9               0.509 0.409       0.327        0.691
    ## 10               0.522 0.442       0.363        0.680

    model_list[7]

    ## [1] "Diagnosis ~ PhonatationTime + PitchVariability + Speechrate + PauseDuration + (1|uID)"

    CVlist2 <- cvms::cross_validate(featurefold, model_list2,
                          family = "binomial",
                          control = glmerControl(
                            optimizer = "nloptwrap",
                            calc.derivs = FALSE,
                            optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8, maxfun = 1000)),
                          REML = FALSE,
                          rm_nc = FALSE
                          )
    CVlist2[,1:4] # model 2 ("Diagnosis ~ PhonatationTime * PitchVariability * PauseDuration + (1|uID)") had the highest accuracy of 0.531 and a pretty balanced sens. and spec.(0.541, 0.522)

    ## # A tibble: 6 x 4
    ##   `Balanced Accuracy`    F1 Sensitivity Specificity
    ##                 <dbl> <dbl>       <dbl>       <dbl>
    ## 1               0.524 0.558      0.562        0.486
    ## 2               0.522 0.432      0.350        0.693
    ## 3               0.510 0.347      0.253        0.768
    ## 4               0.513 0.452      0.384        0.642
    ## 5               0.510 0.101      0.0547       0.965
    ## 6               0.526 0.379      0.280        0.772

    model_list2[2]

    ## [1] "Diagnosis ~ PhonatationTime * PitchVariability * PauseDuration + (1|uID)"

    CVlist3 <- cvms::cross_validate(featurefold, model_list3,
                          family = "binomial",
                          control = glmerControl(
                            optimizer = "nloptwrap",
                            calc.derivs = FALSE,
                            optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8, maxfun = 1000)),
                          REML = FALSE,
                          rm_nc = FALSE
                          )

    CVlist3[,1:4] # model 3 ("Diagnosis ~ PhonatationTime * PitchVariability * Speechrate * PauseDuration + (1|uID)") had the highest accuracy of 0.517 and sens. and spec. somewhat unbalanced (0.301, 0.733)

    ## # A tibble: 3 x 4
    ##   `Balanced Accuracy`    F1 Sensitivity Specificity
    ##                 <dbl> <dbl>       <dbl>       <dbl>
    ## 1               0.496 0.446       0.387       0.604
    ## 2               0.504 0.361       0.272       0.736
    ## 3               0.518 0.374       0.279       0.756

    model_list2[3]

    ## [1] "Diagnosis ~ PhonatationTime * PitchVariability * Speechrate * PauseDuration + (1|uID)"

    #Cross-validation of the best combinations chosen from highest accuracy and somewhat balanced sensitivity and specificity
    CV2 <- cvms::cross_validate(featurefold, "Diagnosis ~ PhonatationTime * PitchVariability * PauseDuration + (1|uID)", 
                           fold_cols = '.folds',
                          family = "binomial",
                          control = glmerControl(
                            optimizer = "nloptwrap",
                            calc.derivs = F,
                            optCtrl=list(xtol_abs=1e-8, ftol_abs=1e-8, maxfun = 1000)),
                          REML = FALSE,
                          rm_nc = FALSE
                          )
    head(CV2)

    ## # A tibble: 1 x 29
    ##   `Balanced Accur…    F1 Sensitivity Specificity `Pos Pred Value`
    ##              <dbl> <dbl>       <dbl>       <dbl>            <dbl>
    ## 1            0.522 0.432       0.350       0.693            0.564
    ## # … with 24 more variables: `Neg Pred Value` <dbl>, AUC <dbl>, `Lower
    ## #   CI` <dbl>, `Upper CI` <dbl>, Kappa <dbl>, MCC <dbl>, `Detection
    ## #   Rate` <dbl>, `Detection Prevalence` <dbl>, Prevalence <dbl>,
    ## #   Predictions <list>, ROC <list>, `Confusion Matrix` <list>,
    ## #   Coefficients <list>, Folds <int>, `Fold Columns` <int>, `Convergence
    ## #   Warnings` <int>, `Singular Fit Messages` <int>, `Other
    ## #   Warnings` <int>, `Warnings and Messages` <list>, Family <chr>,
    ## #   Link <chr>, Dependent <chr>, Fixed <chr>, Random <chr>

    # Plot results for binomial model
    cv_plot(CV2, type ='ROC')

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-6-1.png)

1.  Try other classification methods (Discriminant Function, Random
    Forest, Support Vector Machine, etc. The package caret provides
    them.)

<!-- -->

    #Load unscaled data
    df <- read.csv("features_unscaled.csv") %>% 
      mutate_at(c("Diagnosis"), as.factor)

    #Choosing the relevant features
    df <- df %>% select('Gender','Diagnosis','Study','PhonatationTime','PitchVariability','Speechrate','PauseDuration','uID')

    ##Paritioning##
    set.seed(5)
    df_list <- partition(df, p = 0.2, cat_col = c("Diagnosis", "Gender", "Study"), id_col = "uID", list_out = T)
    df_test = df_list[[1]]
    df_train = df_list[[2]]

    #Removing ID column
    df_test <- df_test %>% 
      select(-uID)
    df_train <- df_train %>% 
      select(-uID)

    #create recipe
    rec <- df_train %>% recipe(Diagnosis ~ .) %>% # defines the outcome
      step_center(all_numeric()) %>% # center numeric predictors
      step_scale(all_numeric()) %>% # scales numeric predictors
      step_corr(all_numeric()) %>% 
      prep(training = df_train)

    train_baked <- juice(rec) # extract df_train from rec

    rec #inspect rec

    ## Data Recipe
    ## 
    ## Inputs:
    ## 
    ##       role #variables
    ##    outcome          1
    ##  predictor          6
    ## 
    ## Training data contained 3036 data points and no missing data.
    ## 
    ## Operations:
    ## 
    ## Centering for Study, PhonatationTime, ... [trained]
    ## Scaling for Study, PhonatationTime, ... [trained]
    ## Correlation filter removed no terms [trained]

    #Applying recepe to a test
    test_baked <- rec %>% bake(df_test)

    #Demonstration of different ways of doing the same process
    juice(rec) 

    ## # A tibble: 3,036 x 7
    ##    Gender Study PhonatationTime PitchVariability Speechrate PauseDuration
    ##    <fct>  <dbl>           <dbl>            <dbl>      <dbl>         <dbl>
    ##  1 M       1.06          -0.812           -0.211     -0.986        -0.969
    ##  2 M       1.06           3.19            -0.401      0.841        -0.500
    ##  3 M       1.06           1.89            -0.366      1.19         -0.268
    ##  4 M       1.06           3.68            -0.381      1.03         -0.454
    ##  5 M       1.06           2.01            -0.368      0.491        -0.128
    ##  6 M       1.06           1.54            -0.282      0.160        -0.166
    ##  7 M       1.06          -0.851           -0.562     -1.65         -0.969
    ##  8 M       1.06          -0.139           -0.366      0.500         0.655
    ##  9 M       1.06           4.88            -0.321      1.09         -0.503
    ## 10 M       1.06           3.19            -0.314      0.471        -0.233
    ## # … with 3,026 more rows, and 1 more variable: Diagnosis <fct>

    rec %>% bake(df_train)

    ## # A tibble: 3,036 x 7
    ##    Gender Diagnosis Study PhonatationTime PitchVariability Speechrate
    ##    <fct>  <fct>     <dbl>           <dbl>            <dbl>      <dbl>
    ##  1 M      0          1.06          -0.812           -0.211     -0.986
    ##  2 M      0          1.06           3.19            -0.401      0.841
    ##  3 M      0          1.06           1.89            -0.366      1.19 
    ##  4 M      0          1.06           3.68            -0.381      1.03 
    ##  5 M      0          1.06           2.01            -0.368      0.491
    ##  6 M      0          1.06           1.54            -0.282      0.160
    ##  7 M      0          1.06          -0.851           -0.562     -1.65 
    ##  8 M      0          1.06          -0.139           -0.366      0.500
    ##  9 M      0          1.06           4.88            -0.321      1.09 
    ## 10 M      0          1.06           3.19            -0.314      0.471
    ## # … with 3,026 more rows, and 1 more variable: PauseDuration <dbl>

    ##Creating models##
    #logistic regression
    log_fit <- 
      logistic_reg() %>%
      set_mode("classification") %>% 
      set_engine("glm") %>%
      fit(Diagnosis ~ . , data = train_baked)

    #support vector machine
    svm_fit <-
      svm_rbf() %>%
      set_mode("classification") %>% 
      set_engine("kernlab") %>%
      fit(Diagnosis ~ . , data = train_baked)

    #random forest
    RanfomForest <-
      rand_forest() %>%
      set_mode("classification") %>% 
      set_engine("randomForest") %>%
      fit(Diagnosis ~ . , data = train_baked)

    #boosted tree
    BoostedTree <-
      boost_tree() %>%
      set_mode("classification") %>% 
      set_engine("xgboost") %>%
      fit(Diagnosis ~ . , data = train_baked)

    ##Applying models to test set##

    #get multiple at once
    test_results <- 
      test_baked %>% 
      select(Diagnosis) %>% 
      mutate(
        log_class = predict(log_fit, new_data = test_baked) %>% 
          pull(.pred_class), #predicting class
        log_prob  = predict(log_fit, new_data = test_baked, type = "prob") %>% 
          pull(.pred_1), #probability outcome
        svm_class = predict(svm_fit, new_data = test_baked) %>% 
          pull(.pred_class), 
        svm_prob  = predict(svm_fit, new_data = test_baked, type = "prob") %>% 
          pull(.pred_1),
        ranfor_class = predict(RanfomForest, new_data = test_baked) %>% 
          pull(.pred_class),
        ranfor_prob  = predict(RanfomForest, new_data = test_baked, type = "prob") %>% 
          pull(.pred_1),
        btree_class = predict(BoostedTree, new_data = test_baked) %>% 
          pull(.pred_class),
        btree_prob  = predict(BoostedTree, new_data = test_baked, type = "prob") %>% 
          pull(.pred_1)
      )
      
    view(test_results)
    ###Performance

    #Performance metrics
    metrics(test_results, truth = Diagnosis, estimate = log_class) %>% view()
    metrics(test_results, truth = Diagnosis, estimate = svm_class) %>% view() #best accuracy (0.6585678)
    metrics(test_results, truth = Diagnosis, estimate = ranfor_class) %>% view()
    metrics(test_results, truth = Diagnosis, estimate = btree_class) %>% view()

    #ROC curve
    #plotting the roc curve:
    test_results %>%
      roc_curve(truth = Diagnosis, log_prob) %>% 
      autoplot()

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-7-1.png)

    test_results %>%
      roc_curve(truth = Diagnosis, svm_prob) %>% 
      autoplot()

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-7-2.png)

    test_results %>%
      roc_curve(truth = Diagnosis, ranfor_prob) %>% 
      autoplot()

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-7-3.png)

    test_results %>%
      roc_curve(truth = Diagnosis, btree_prob) %>% 
      autoplot()

![](A3_P2_DiagnosingSchizophrenia_instructions-1_files/figure-markdown_strict/unnamed-chunk-7-4.png)
