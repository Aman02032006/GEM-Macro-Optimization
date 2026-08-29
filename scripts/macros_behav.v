/* Behavioural models of the three intercepted primitives.
 *
 * These exist ONLY to build the Deliverable D baseline: read these instead of
 * scripts/macros_bb.v and Yosys shreds each macro into AIG gates, which is
 * exactly what unmodified GEM does. The macro build reads the blackbox
 * declarations instead and leaves the cells intact.
 *
 * Semantics are transcribed from the problem statement and match the verified
 * golden models in src/models/*.cu, so baseline and macro builds must produce
 * bit-identical waveforms -- which is itself a check on the whole premise.
 */

module CARRY4 (
    output [3:0] CO,
    output [3:0] O,
    input        CI,
    input        CYINIT,
    input  [3:0] DI,
    input  [3:0] S
);
    wire [4:0] C;
    assign C[0] = CI | CYINIT;
    assign C[1] = S[0] ? C[0] : DI[0];
    assign C[2] = S[1] ? C[1] : DI[1];
    assign C[3] = S[2] ? C[2] : DI[2];
    assign C[4] = S[3] ? C[3] : DI[3];
    assign O    = S ^ C[3:0];
    assign CO   = C[4:1];
endmodule

module SRLC32E (
    output       Q,
    output       Q31,
    input  [4:0] A,
    input        CE,
    input        CLK,
    input        D
);
    reg [31:0] sr = 32'b0;
    always @(posedge CLK) begin
        if (CE) sr <= {sr[30:0], D};
    end
    assign Q   = sr[A];
    assign Q31 = sr[31];
endmodule

/* AD = A + D wrapped to 27 bits, M = AD*B exact in 45, P += M mod 2^48. */
module DSP48E2_MAC_PREADD (
    output [47:0] P,
    input  [26:0] A,
    input  [17:0] B,
    input  [47:0] C,
    input  [26:0] D,
    input         CEP,
    input         CLK
);
    reg  signed [47:0] preg = 48'b0;
    wire signed [26:0] ad = $signed(A) + $signed(D);
    wire signed [44:0] m  = $signed(ad) * $signed(B);
    always @(posedge CLK) begin
        if (CEP) preg <= preg + {{3{m[44]}}, m};
    end
    assign P = preg;
endmodule
