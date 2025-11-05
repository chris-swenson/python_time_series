# ARIMA trend=
# no integration -> trend='c' for mean estimate called "constant"
# one integration -> trend='n' for no mean/constant, 'c' is not allowed
# both integrations -> ''

# no integration, defaults to trend='c'
trend = [i+1 for i in range(0, len(flow))]
print('Default')
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12)).fit().summary().tables[1]) # trend='c', mean
print('\ntrend=n')
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12), trend='n').fit().summary().tables[1]) # no mean/constant
print('\ntrend=t')
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12), trend='t').fit().summary().tables[1]) # trend
print('\ntrend=n and exog=trend')
print(tsa.ARIMA(flow, trend, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12), trend='n').fit().summary().tables[1]) # trend
print('\ntrend=c')
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 0, 1, 12), trend='c').fit().summary().tables[1]) # mean

# nonseasonal integration, defaults to no mean/constant
print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 0, 1, 12)).fit().summary().tables[1]) # trend='n', no mean/constant
print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 0, 1, 12), trend='n').fit().summary().tables[1]) # no mean/constant
print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 0, 1, 12), trend='t').fit().summary().tables[1]) # mean
#print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 0, 1, 12), trend='c').fit().summary().tables[1]) # error

# seasonal integration, defaults to no mean/constant
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12)).fit().summary().tables[1]) # trend='n', no mean/constant
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend='n').fit().summary().tables[1]) # no mean/constant
print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend='t').fit().summary().tables[1]) # mean
#print(tsa.ARIMA(flow, order=(1, 0, 0), seasonal_order=(0, 1, 1, 12), trend='c').fit().summary().tables[1]) # error

# both integrations, defaults to no mean, no constant
print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12)).fit().summary().tables[1]) # trend='n', no mean/constant
print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), trend='n').fit().summary().tables[1]) # no mean/constant
#print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), trend='t').fit().summary().tables[1]) # error
#print(tsa.ARIMA(flow, order=(1, 1, 0), seasonal_order=(0, 1, 1, 12), trend='c').fit().summary().tables[1]) # error


def sarima(
    xdata, p, d, q, P = 0, D = 0, Q = 0, S = -1, 
    no_constant = FALSE, 
    xreg = None, fixed = None, 
    tol = sqrt(.Machine$double.eps),
    **kwargs
):
    
    # removed the fixed= argument as it doesn't appear supported
    # this option would set some parameters as static

    arima = tsa.arima.ARIMA
    #trans = ifelse(is.null(fixed), TRUE, FALSE)
    trans = True if fixed == None else False
    xdata = np.array(xdata)
    n = len(xdata)
    
    # no external regressors
    if (xreg == None):
        
        constant = 1:n
        xmean = rep(1, n)
        
        if (no_constant == True):
            xmean = None
            
        # no differences
        if (d == 0 & D == 0):
            fitit = arima(
                xdata, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = S), 
                xreg = xmean, include.mean = False, fixed = fixed, trans = trans, 
                optim.control = list(trace = trc, REPORT = 1, reltol = tol),
                **kwargs
            )
        }
        
        # at least one difference
        else if (xor(d == 1, D == 1) & no_constant == FALSE):
            fitit = arima(
                xdata, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = S), 
                xreg = constant, fixed = fixed, trans = trans, 
                optim.control = list(trace = trc, REPORT = 1, reltol = tol),
                **kwargs
            )
        
        # all others
        else:
            fitit = arima(
                xdata, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = S), 
                include.mean = !no_constant, fixed = fixed, trans = trans, 
                optim.control = list(trace = trc, REPORT = 1, reltol = tol),
                **kwargs
            )
    
    # external regressors
    if (xreg != None):
        fitit = stats::arima(
            xdata, order = c(p, d, q), seasonal = list(order = c(P, D, Q), period = S), 
            xreg = xreg, fixed = fixed, trans = trans, 
            optim.control = list(trace = trc, REPORT = 1, reltol = tol),
            **kwargs
        )
    
    # Split DETAILS section out

    if (is.null(fixed)) {
        coefs = fitit$coef
    }
    else {
        coefs = fitit$coef[is.na(fixed)]
    }
    
    k = length(coefs)
    n = fitit$nobs
    dfree = n - k
    t.value = coefs/sqrt(diag(fitit$var.coef))
    p.two = stats::pf(t.value^2, df1 = 1, df2 = dfree, lower.tail = FALSE)
    ttable = cbind(Estimate = coefs, SE = sqrt(diag(fitit$var.coef)), t.value, p.value = p.two)
    ttable = round(ttable, 4)
    BIC = stats::BIC(fitit)/n
    AIC = stats::AIC(fitit)/n
    AICc = (n * AIC + ((2 * k^2 + 2 * k)/(n - k - 1)))/n
    list(fit = fitit, degrees_of_freedom = dfree, ttable = ttable, AIC = AIC, AICc = AICc, BIC = BIC)
﻿
