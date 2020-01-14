module Trace

export @trace, @traceon, @traceoff, @gettrace

_trace_status = :off
_trace_list = Vector{Any}()

function _traceon()
    global _trace_status = :on
end    

function _traceoff()
    global _trace_status = :off
end    

function _traceclear()
    global _trace_list = Vector{Any}()
end

macro trace(expr)
    body = quote
        if Trace._trace_status == :on
            push!(Trace._trace_list, $expr)
        else
            $expr
        end
    end
    esc(body)
end

macro traceon(x)
    body = quote
        Trace._traceclear()
        Trace._traceon()
        _trace_f = $x
        Trace._traceoff()
        _trace_f
    end
    esc(body)
end

macro traceoff(x)
    body = quote
        Trace._traceoff()
        _trace_f = $x
        Trace._traceon()
        _trace_f
    end
    esc(body)
end

macro gettrace()
    esc(:(Trace._trace_list))
end

end
