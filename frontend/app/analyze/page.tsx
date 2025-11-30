"use client";

import { useState, useEffect } from "react";
import { Cormorant_Garamond } from "next/font/google";

const cormorantGaramond = Cormorant_Garamond({
  weight: ["300", "400", "500", "600", "700"],
  style: ["italic", "normal"],
  subsets: ["latin"],
  variable: "--font-cormorant-garamond",
});

const API_URL = "http://localhost:8000";

interface FormData {
  latitude: number;
  longitude: number;
  neighbourhood: string;
  room_type: string;
  property_type: string;
  accommodates: number;
  bedrooms: number;
  beds: number;
  bathrooms: number;
  amenities: string[];
  host_experience_days: number;
  num_listings: number;
  review_scores_rating: number;
  review_scores_accuracy: number;
  review_scores_cleanliness: number;
  review_scores_checkin: number;
  review_scores_communication: number;
  review_scores_location: number;
  review_scores_value: number;
  number_of_reviews: number;
}

interface Options {
  neighbourhoods: string[];
  room_types: string[];
  property_types: string[];
  amenities: string[];
}

const defaultFormData: FormData = {
  latitude: 52.52,
  longitude: 13.405,
  neighbourhood: "Mitte",
  room_type: "Entire home/apt",
  property_type: "Entire rental unit",
  accommodates: 2,
  bedrooms: 1,
  beds: 1,
  bathrooms: 1,
  amenities: [],
  host_experience_days: 365,
  num_listings: 1,
  review_scores_rating: 4.5,
  review_scores_accuracy: 4.5,
  review_scores_cleanliness: 4.5,
  review_scores_checkin: 4.5,
  review_scores_communication: 4.5,
  review_scores_location: 4.5,
  review_scores_value: 4.5,
  number_of_reviews: 10,
};

export default function AnalyzePage() {
  const [step, setStep] = useState(1);
  const [formData, setFormData] = useState<FormData>(defaultFormData);
  const [options, setOptions] = useState<Options | null>(null);
  const [prediction, setPrediction] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [amenityModalOpen, setAmenityModalOpen] = useState(false);
  const [mapUrl, setMapUrl] = useState(`${API_URL}/map`);

  useEffect(() => {
    fetch(`${API_URL}/options`)
      .then((res) => res.json())
      .then((data) => setOptions(data))
      .catch((err) => console.error("Failed to load options:", err));
  }, []);

  const updateField = <K extends keyof FormData>(
    field: K,
    value: FormData[K]
  ) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const toggleAmenity = (amenity: string) => {
    setFormData((prev) => ({
      ...prev,
      amenities: prev.amenities.includes(amenity)
        ? prev.amenities.filter((a) => a !== amenity)
        : [...prev.amenities, amenity],
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formData),
      });
      const data = await response.json();
      setPrediction(data.predicted_price);
      setMapUrl(
        `${API_URL}/map?lat=${formData.latitude}&lng=${formData.longitude}&price=${data.predicted_price}`
      );
    } catch (err) {
      console.error("Prediction failed:", err);
    }
    setLoading(false);
  };

  const inputClass =
    "w-full rounded-xl border border-black/30 bg-white/80 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-black/60 text-sm";
  const labelClass = "block text-sm font-semibold mb-1";

  const renderStep = () => {
    switch (step) {
      case 1:
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-bold mb-4">Step 1: Location</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Latitude</label>
                <input
                  type="number"
                  step="0.0001"
                  className={inputClass}
                  value={formData.latitude}
                  onChange={(e) =>
                    updateField("latitude", parseFloat(e.target.value) || 0)
                  }
                />
              </div>
              <div>
                <label className={labelClass}>Longitude</label>
                <input
                  type="number"
                  step="0.0001"
                  className={inputClass}
                  value={formData.longitude}
                  onChange={(e) =>
                    updateField("longitude", parseFloat(e.target.value) || 0)
                  }
                />
              </div>
            </div>
            <div>
              <label className={labelClass}>Neighbourhood</label>
              <select
                className={inputClass}
                value={formData.neighbourhood}
                onChange={(e) => updateField("neighbourhood", e.target.value)}
              >
                {options?.neighbourhoods.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-bold mb-4">Step 2: Property Details</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Room Type</label>
                <select
                  className={inputClass}
                  value={formData.room_type}
                  onChange={(e) => updateField("room_type", e.target.value)}
                >
                  {options?.room_types.map((rt) => (
                    <option key={rt} value={rt}>
                      {rt}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className={labelClass}>Property Type</label>
                <select
                  className={inputClass}
                  value={formData.property_type}
                  onChange={(e) => updateField("property_type", e.target.value)}
                >
                  {options?.property_types.map((pt) => (
                    <option key={pt} value={pt}>
                      {pt}
                    </option>
                  ))}
                </select>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Guests (Accommodates)</label>
                <input
                  type="number"
                  min={1}
                  className={inputClass}
                  value={formData.accommodates}
                  onChange={(e) =>
                    updateField("accommodates", parseInt(e.target.value) || 1)
                  }
                />
              </div>
              <div>
                <label className={labelClass}>Bedrooms</label>
                <input
                  type="number"
                  min={0}
                  step={1}
                  className={inputClass}
                  value={formData.bedrooms}
                  onChange={(e) =>
                    updateField("bedrooms", parseFloat(e.target.value) || 0)
                  }
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Beds</label>
                <input
                  type="number"
                  min={0}
                  step={1}
                  className={inputClass}
                  value={formData.beds}
                  onChange={(e) =>
                    updateField("beds", parseFloat(e.target.value) || 0)
                  }
                />
              </div>
              <div>
                <label className={labelClass}>Bathrooms</label>
                <input
                  type="number"
                  min={0}
                  step={0.5}
                  className={inputClass}
                  value={formData.bathrooms}
                  onChange={(e) =>
                    updateField("bathrooms", parseFloat(e.target.value) || 0)
                  }
                />
              </div>
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-bold mb-4">Step 3: Amenities</h2>
            <p className="text-sm text-gray-600 mb-2">
              Selected: {formData.amenities.length} amenities
            </p>
            <button
              type="button"
              onClick={() => setAmenityModalOpen(true)}
              className="w-full px-4 py-3 bg-white/80 border border-black/30 rounded-xl text-left hover:bg-white transition-colors"
            >
              <span className="font-semibold">Click to select amenities</span>
              {formData.amenities.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1">
                  {formData.amenities.slice(0, 5).map((a) => (
                    <span
                      key={a}
                      className="text-xs bg-black/10 px-2 py-1 rounded"
                    >
                      {a}
                    </span>
                  ))}
                  {formData.amenities.length > 5 && (
                    <span className="text-xs bg-black/10 px-2 py-1 rounded">
                      +{formData.amenities.length - 5} more
                    </span>
                  )}
                </div>
              )}
            </button>
          </div>
        );

      case 4:
        return (
          <div className="space-y-4">
            <h2 className="text-xl font-bold mb-4">Step 4: Host & Reviews</h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={labelClass}>Host Experience (days)</label>
                <input
                  type="number"
                  min={0}
                  className={inputClass}
                  value={formData.host_experience_days}
                  onChange={(e) =>
                    updateField(
                      "host_experience_days",
                      parseInt(e.target.value) || 0
                    )
                  }
                />
              </div>
              <div>
                <label className={labelClass}>Number of Listings</label>
                <input
                  type="number"
                  min={1}
                  className={inputClass}
                  value={formData.num_listings}
                  onChange={(e) =>
                    updateField("num_listings", parseInt(e.target.value) || 1)
                  }
                />
              </div>
            </div>
            <div>
              <label className={labelClass}>Number of Reviews</label>
              <input
                type="number"
                min={0}
                className={inputClass}
                value={formData.number_of_reviews}
                onChange={(e) =>
                  updateField(
                    "number_of_reviews",
                    parseInt(e.target.value) || 0
                  )
                }
              />
            </div>
            <div className="border-t border-black/20 pt-4 mt-4">
              <h3 className="font-semibold mb-3">Review Scores (1-5)</h3>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { key: "review_scores_rating", label: "Overall Rating" },
                  { key: "review_scores_accuracy", label: "Accuracy" },
                  { key: "review_scores_cleanliness", label: "Cleanliness" },
                  { key: "review_scores_checkin", label: "Check-in" },
                  { key: "review_scores_communication", label: "Communication" },
                  { key: "review_scores_location", label: "Location" },
                  { key: "review_scores_value", label: "Value" },
                ].map(({ key, label }) => (
                  <div key={key}>
                    <label className="block text-xs font-medium mb-1">
                      {label}
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={5}
                      step={0.1}
                      className={inputClass}
                      value={formData[key as keyof FormData] as number}
                      onChange={(e) =>
                        updateField(
                          key as keyof FormData,
                          parseFloat(e.target.value) || 1
                        )
                      }
                    />
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

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
          Analyze Your <span className="font-bold italic">Airbnb</span> Potential
        </h1>
        <p className="text-center text-sm text-gray-600 mb-8">
          Fill in your property details to get a price prediction
        </p>

        <div className="grid gap-8 md:grid-cols-3 flex-1">
          <div className="md:col-span-2 rounded-2xl overflow-hidden shadow-lg">
            <iframe
              src={mapUrl}
              className="w-full h-full min-h-[500px] border-0"
              title="Property Map"
            />
          </div>

          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 flex flex-col">
            <div className="flex justify-between mb-6">
              {[1, 2, 3, 4].map((s) => (
                <button
                  key={s}
                  onClick={() => setStep(s)}
                  className={`w-10 h-10 rounded-full font-bold transition-all ${
                    step === s
                      ? "bg-black text-white"
                      : step > s
                      ? "bg-green-500 text-white"
                      : "bg-gray-200 text-gray-600"
                  }`}
                >
                  {step > s ? "✓" : s}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto">{renderStep()}</div>

            <div className="mt-6 space-y-3">
              <div className="flex gap-2">
                {step > 1 && (
                  <button
                    type="button"
                    onClick={() => setStep(step - 1)}
                    className="flex-1 px-4 py-2 bg-gray-200 text-black rounded-full font-semibold hover:bg-gray-300 transition-all"
                  >
                    Back
                  </button>
                )}
                {step < 4 && (
                  <button
                    type="button"
                    onClick={() => setStep(step + 1)}
                    className="flex-1 px-4 py-2 bg-black text-white rounded-full font-semibold hover:bg-gray-800 transition-all"
                  >
                    Next
                  </button>
                )}
              </div>

              {step === 4 && (
                <button
                  type="button"
                  onClick={handlePredict}
                  disabled={loading}
                  className="w-full px-6 py-3 bg-black text-white rounded-full font-semibold text-lg border-2 border-black transition-all shadow-xl hover:shadow-black/60 hover:scale-105 active:scale-100 disabled:opacity-50"
                >
                  {loading ? "Predicting..." : "Predict Price"}
                </button>
              )}

              {prediction !== null && (
                <div className="mt-4 p-4 bg-green-100 rounded-xl text-center">
                  <p className="text-sm text-gray-600">Predicted Nightly Price</p>
                  <p className="text-3xl font-bold text-green-700">
                    ${prediction.toFixed(2)}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {amenityModalOpen && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl p-6 max-w-lg w-full max-h-[80vh] overflow-y-auto">
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-xl font-bold">Select Amenities</h3>
              <button
                onClick={() => setAmenityModalOpen(false)}
                className="text-2xl hover:text-gray-600"
              >
                &times;
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2">
              {options?.amenities.map((amenity) => (
                <label
                  key={amenity}
                  className={`flex items-center p-2 rounded-lg cursor-pointer transition-colors ${
                    formData.amenities.includes(amenity)
                      ? "bg-black text-white"
                      : "bg-gray-100 hover:bg-gray-200"
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={formData.amenities.includes(amenity)}
                    onChange={() => toggleAmenity(amenity)}
                    className="sr-only"
                  />
                  <span className="text-sm">{amenity}</span>
                </label>
              ))}
            </div>
            <button
              onClick={() => setAmenityModalOpen(false)}
              className="mt-4 w-full px-4 py-2 bg-black text-white rounded-full font-semibold"
            >
              Done ({formData.amenities.length} selected)
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
