`timescale 1ns / 1ps

/* Macro-rich micro-benchmark for the GEM word-level macro path.
 *
 * Deliberately exercises all three intercepted primitives at once, which the
 * MIPS core cannot: it has no multiplier and no shift-register chains, so it
 * only ever produces CARRY4.
 *
 *   DSP48E2_MAC_PREADD : one per lane, accumulating (A + D) * B
 *   SRLC32E            : a 2-deep cascade per lane, Q31 -> D
 *   CARRY4             : from the wide adders below, via techmap
 *
 * The macros are instantiated by name because that is what the problem
 * statement describes ("intercept the explicitly named DSP48E2, CARRY4 and
 * SRLC32E primitives"). The DSP is written in its mangled form directly, since
 * cell parameters do not survive netlistdb -- see src/macros.rs.
 *
 * Every lane is perturbed by its index so the optimiser cannot merge lanes
 * into one.
 */
module macro_bench #(
    parameter LANES = 16
) (
    input                    clk,
    input                    rst,
    input  [26:0]            a_in,
    input  [17:0]            b_in,
    input  [26:0]            d_in,
    input                    srl_d,
    input                    srl_ce,
    input  [4:0]             srl_a,
    input  [31:0]            add_a,
    input  [31:0]            add_b,
    input                    add_cin,
    output [31:0]            mac_lo,
    output [31:0]            sum_out,
    output [LANES-1:0]       delay_out,
    output [LANES-1:0]       delay_tail
);

    reg [26:0] a_q;
    reg [17:0] b_q;
    reg [26:0] d_q;
    reg [31:0] adda_q;
    reg [31:0] addb_q;

    always @(posedge clk) begin
        if (rst) begin
            a_q    <= 27'b0;
            b_q    <= 18'b0;
            d_q    <= 27'b0;
            adda_q <= 32'b0;
            addb_q <= 32'b0;
        end
        else begin
            a_q    <= a_in;
            b_q    <= b_in;
            d_q    <= d_in;
            adda_q <= add_a;
            addb_q <= add_b;
        end
    end

    wire [47:0] p_lane   [0:LANES-1];
    wire [31:0] sum_lane [0:LANES-1];

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : lane

            /* ---- DSP48E2: accumulate (A + D) * B ------------------------ */
            wire [26:0] a_l = a_q ^ i[26:0];
            wire [26:0] d_l = d_q + i[26:0];
            wire [17:0] b_l = b_q ^ i[17:0];

            DSP48E2_MAC_PREADD dsp (
                .CLK (clk),
                .A   (a_l),
                .B   (b_l),
                .C   (48'b0),
                .D   (d_l),
                .CEP (1'b1),
                .P   (p_lane[i])
            );

            /* ---- SRLC32E: 2-deep cascade, Q31 -> D ---------------------- */
            wire srl0_q, srl0_q31, srl1_q, srl1_q31;

            SRLC32E srl0 (
                .CLK (clk),
                .D   (srl_d ^ i[0]),
                .CE  (srl_ce),
                .A   (srl_a),
                .Q   (srl0_q),
                .Q31 (srl0_q31)
            );

            SRLC32E srl1 (
                .CLK (clk),
                .D   (srl0_q31),
                .CE  (srl_ce),
                .A   (srl_a),
                .Q   (srl1_q),
                .Q31 (srl1_q31)
            );

            assign delay_out[i]  = srl0_q;
            assign delay_tail[i] = srl1_q;

            /* ---- wide adder: techmap turns this into a CARRY4 chain ----- */
            assign sum_lane[i] = adda_q + (addb_q ^ {28'b0, i[3:0]})
                               + {31'b0, add_cin};
        end
    endgenerate

    /* Reduce over EVERY lane. Reading only the first and last would let the
     * optimiser prune the other 14 lanes -- which it duly did on the first
     * attempt, leaving 2 DSPs instead of 16. */
    reg [47:0] mac_red;
    reg [31:0] sum_red;
    integer j;
    always @(*) begin
        mac_red = 48'b0;
        sum_red = 32'b0;
        for (j = 0; j < LANES; j = j + 1) begin
            mac_red = mac_red ^ p_lane[j];
            sum_red = sum_red ^ sum_lane[j];
        end
    end

    reg [47:0] mac_acc;
    reg [31:0] sum_acc;
    always @(posedge clk) begin
        if (rst) begin
            mac_acc <= 48'b0;
            sum_acc <= 32'b0;
        end
        else begin
            mac_acc <= mac_red;
            sum_acc <= sum_red;
        end
    end

    assign mac_lo  = mac_acc[31:0];
    assign sum_out = sum_acc;

endmodule
