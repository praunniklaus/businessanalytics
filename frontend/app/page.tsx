import Link from "next/link";
import { Cormorant_Garamond } from "next/font/google";

const cormorantGaramond = Cormorant_Garamond({
  weight: ["300", "400", "500", "600", "700"],
  style: ["italic", "normal"],
  subsets: ["latin"],
  variable: "--font-cormorant-garamond",
});

export default function Home() {
  return (
    <div className="relative min-h-screen bg-cover bg-bottom bg-no-repeat" style={{ backgroundImage: "url('/img/bg.png')" }}>
      <div className={`relative z-10 flex flex-col items-center justify-end pb-[45vh] min-h-screen ${cormorantGaramond.className}`}>
        <h1 className="text-6xl italic text-center mb-10">
          See Your <span className="font-bold italic">Airbnb</span> Earning Potential.
        </h1>
        <Link href="/analyze">
          <button className="px-16 py-5 bg-transparent text-black rounded-full font-semibold text-2xl border-2 border-black transition-all shadow-2xl hover:shadow-black/60 hover:scale-105 active:scale-100">
            Analyze now
          </button>
        </Link>
      </div>
    </div>
  );
}

