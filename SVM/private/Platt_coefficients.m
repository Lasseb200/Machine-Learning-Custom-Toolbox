function [A_Platt, B_Platt] = Platt_coefficients(F_training,yr)
    t = (yr + 1)/2;
    negLogL = @(ab) -sum(t.*log(1./(1+exp(ab(1)*F_training+ab(2))))+(1-t).*log(1-1./(1+exp(ab(1)*F_training+ab(2)))));
    ab0 = [0 0];
    opt = optimoptions('fminunc','Display','off','Algorithm','quasi-newton');
    ab_optimal = fminunc(negLogL, ab0, opt);
    A_Platt = ab_optimal(1);
    B_Platt = ab_optimal(2);
end
