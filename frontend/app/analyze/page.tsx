import { Cormorant_Garamond } from "next/font/google";

const cormorantGaramond = Cormorant_Garamond({
  weight: ["300", "400", "500", "600", "700"],
  style: ["italic", "normal"],
  subsets: ["latin"],
  variable: "--font-cormorant-garamond",
});

export default function AnalyzePage() {
  return (
    <div
      className="relative min-h-screen bg-cover bg-bottom bg-no-repeat"
      style={{ backgroundImage: "url('/img/bg.png')" }}
    >
      <div className="absolute inset-0 bg-[lightblue]/30"></div>
      <div
        className={`relative z-10 flex flex-col min-h-screen px-8 py-12 ${cormorantGaramond.className}`}
      >
        <h1 className="text-4xl md:text-5xl italic text-center mb-2">
          Analyze Your <span className="font-bold italic">Airbnb</span>{" "}
          Potential
        </h1>
        <p className="text-center text-sm text-gray-600 mb-8">
          *Placeholders – components will be integrated soon
        </p>

        <div className="grid gap-8 md:grid-cols-3 flex-1">
          <div className="md:col-span-2 border-2 border-dashed border-black/40 rounded-2xl bg-white/40 flex flex-col items-center justify-center text-center text-black/70 p-8">
            <div className="text-lg font-semibold mb-2">Heatmap placeholder</div>
            (pricing & demand visualization)
          </div>

          <form className="space-y-4 bg-white/70 backdrop-blur-sm rounded-2xl p-6">
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Location
                </label>
                <input
                  type="text"
                  className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60"
                  placeholder="City, neighborhood"
                />
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold mb-1">
                    Bedrooms
                  </label>
                  <input
                    type="number"
                    min={0}
                    className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60"
                  />
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1">
                    Guests
                  </label>
                  <input
                    type="number"
                    min={1}
                    className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60"
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-semibold mb-1">
                    Property type
                  </label>
                  <select className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60">
                    <option>Entire place</option>
                    <option>Private room</option>
                    <option>Shared room</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-semibold mb-1">
                    Nightly price target (€)
                  </label>
                  <input
                    type="number"
                    min={0}
                    className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60"
                  />
                </div>
              </div>
              <div>
                <label className="block text-sm font-semibold mb-1">
                  Expected occupancy (%)
                </label>
                <input
                  type="number"
                  min={0}
                  max={100}
                  className="w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60"
                  placeholder="e.g. 70"
                />
              </div>
              <button
                type="submit"
                className="mt-4 w-full px-6 py-3 bg-black text-white rounded-full font-semibold text-lg border-2 border-black transition-all shadow-xl hover:shadow-black/60 hover:scale-105 active:scale-100"
              >
                Predict earnings
              </button>
            </form>
        </div>
      </div>
    </div>
  );
}


