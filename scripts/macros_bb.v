(* blackbox *)
module CARRY4 (
    output [3:0] CO,
    output [3:0] O,
    input        CI,
    input        CYINIT,
    input  [3:0] DI,
    input  [3:0] S
);
endmodule

(* blackbox *)
module SRLC32E (
    output       Q,
    output       Q31,
    input  [4:0] A,
    input        CE,
    input        CLK,
    input        D
);
endmodule

(* blackbox *)
module DSP48E2_MAC_PREADD (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule

(* blackbox *)
module DSP48E2_MAC (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule

(* blackbox *)
module DSP48E2_MULT_PREADD (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule

(* blackbox *)
module DSP48E2_MULT (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule

(* blackbox *)
module DSP48E2_BYPASS (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule

(* blackbox *)
module DSP48E2_BYPASS_PREADD (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
endmodule
