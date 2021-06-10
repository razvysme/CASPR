function out = Normalize(in, ampMax)
%%     By YI-WEN CHEN, 2019 / Jarvus Studio
%     Scale speech by its peak value
% 
%     Input Parameters : 
%       in       Input speech
%       ampMax   Expected peak value (0 ~ 1)
%     Output Parameters : enhanced speech  
%       out      Scaled speech
%% M 
    out = zeros(length(in),1);
    if( ampMax > 1 || ampMax < 0 )
        fprintf('(ampMax) out of bound.');
    else
        if max(in) > abs(min(in))
            out = in*(ampMax/max(in));
        else
            out = in*((-ampMax)/min(in));
        end
    end
end