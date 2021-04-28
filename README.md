# Assigment3
  This assignment uses some algorithms of machine learning to estimate fund positions.Fund position refers to the proportion of stock assets held by the fund to the fund's assets. As an important institutional investor in the A-share market, public funds have always received market attention for their shareholding trends. 
  
  On the one hand, out of recognition of the investment ability of fund managers, stock investors generally believe that the position changes of public funds reflect key information such as changes in market investment sentiment; on the other hand, fund investors will also pay attention to the funds they hold at any time. Position changes to assist your own investment decisions. However, public funds only disclose their asset allocation at the end of each quarter, which creates a relative information asymmetry between investors and fund managers. Therefore, the measurement and research of fund positions has become a meaningful work.
  
  The current common fund position measurement methods are mainly based on the traditional index simulation method, that is, the use of fund net value data and index daily data for regression calculation. Fund income comes from the price changes of assets, so the linear relationship between the changes in the net value of the fund and the changes in the net value of stocks corresponds to the proportion of stock assets in the net value of the fund. In the theoretical sense, the index simulation method is to use the change of stock industry index to simulate the change of the fund's equity net value. 
  
  According to market experience, investment managers tend to diversify investment, that is, purchase stocks of different industries or with different characteristics to diversify risks, and the specific types of assets held depend on the personal judgment and preference of investment managers. Therefore, this article selects a group of composite indexes representing different investment styles for weighted synthesis. Following this traditional calculation method, several different regression methods are adopted to calculate the position of the partial stock hybrid fund and compare with the real results to evaluate the pros and cons of several methods.
  
  In the copied research report, the methods involved are linear regression, pca regression, ridge regression and lasso regression. On this basis, I tried to use the quadratic programming method and achieved good results.
  
  I optimized the length of the regression time window and some model parameters by traversing the search method, and finally concluded that ridge regression is the best method for comprehensive results.
  
  ![image](https://user-images.githubusercontent.com/80148045/116342775-c2264680-a815-11eb-9eac-8e869d912935.png)
  
  ![image](https://user-images.githubusercontent.com/80148045/116342980-1d583900-a816-11eb-8ef0-32f3671e3b04.png)

![image](https://user-images.githubusercontent.com/80148045/116343054-45e03300-a816-11eb-9438-a6ae60c69faa.png)

