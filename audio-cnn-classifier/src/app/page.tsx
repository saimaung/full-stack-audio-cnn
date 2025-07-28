"use client"

import { useState } from "react";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";

interface Prediction {
  class: string;
  confidence: number;
}

interface ConvLayerData {
  shape: number[];
  values: number[][];
}

interface VisualizationData {
  [layername: string]: ConvLayerData;
}

interface WaveFormData {
  values: number[];
  sampleRate: number;
  duration: number;
}

interface APIResponse {
  predictions: Prediction[];
  visualizations: VisualizationData;
  input_spectrogram: ConvLayerData;
  waveform: WaveFormData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  door_wood_knock: "🚪",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  sneezing: "🤧",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  door_wood_creaks: "🚪",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  can_opening: "🥫",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  washing_machine: "🧺",
  train: "🚂",
  hen: "🐔",
  wind: "💨",
  laughing: "😂",
  vacuum_cleaner: "🧹",
  church_bells: "🔔",
  insects: "🦟",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  crow: "🐦‍⬛",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
  hand_saw: "🪚",
};

const getEmojiForClass = (className: string): string => {
  return ESC50_EMOJI_MAP[className] || "🔊";
}

function splitConvLayers(visualization: VisualizationData) {
  // top layer data
  const main: [string, ConvLayerData][] = [];
  const internals: Record<string, [string, ConvLayerData][]> = {};
  for (const [name, data] of Object.entries(visualization)) {
    if (!name.includes(".")) {
      main.push([name, data]);
    } else {
      const [parent] = name.split(".");
      if (parent === undefined) {
        continue;
      }
      if (!internals[parent]) {
        internals[parent] = [];
      }
      internals[parent].push([name, data]);
    }
  }
  return { main, internals };
}

export default function HomePage() {
  const [vizData, setVizData] = useState<APIResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    // only handle single file
    const file = event.target.files?.[0];
    if (!file) return;
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);

    const fileReader = new FileReader();
    // array buffer is raw binary data
    fileReader.readAsArrayBuffer(file);
    fileReader.onload = async () => {
      try {
        // binary in string format
        const arrayBuffer = fileReader.result as ArrayBuffer;
        // convert array buffer to base64 string to send it to the endpoint
        const base64String = btoa(
          // Uint8Array is an array of bytes
          // Then loop through each byte and convert it to a character
          // Then join all characters into a string
          new Uint8Array(arrayBuffer).reduce((data, byte) => data + String.fromCharCode(byte), "")
        );
        const response = await fetch("https://saimaung--audiocnn-inference-audioclassfier-inference.modal.run", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ audio_data: base64String }),
        });

        if (!response.ok) {
          throw new Error(`API Call Error: ${response.statusText}`);
        }
        const data: APIResponse = await response.json();
        setVizData(data);
      } catch (error) {
        setError(error instanceof Error ? error.message : "An unknown error occurred.");
      } finally {
        setIsLoading(false);
      }
    }
    fileReader.onerror = () => {
      setError("Failed to read file.");
      setIsLoading(false);
    }
  }

  const { main, internals } = vizData ? vizData.visualizations : { main: [], internals: {} };
  return (
    <main className="min-h-screen pg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="margin-b-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            Audio CNN Classifier
          </h1>
          <p className="mb-8 text-md text-stone-600">
            Upload a WAV file to see Model's predictions and feature maps
          </p>
          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav"
                id="file-upload"
                disabled={isLoading}
                onChange={handleFileChange}
                className="absolute inset-0 w-full cursor-pointer opacity-0"
              />
              <Button
                disabled={isLoading}
                variant="outline"
                size="lg"
                className="border-stone-300">
                {isLoading ? "Processing..." : "Upload a WAV file"}
              </Button>
            </div>
            {fileName && (
              <Badge
                variant="secondary"
                className="mt-4 bg-stone-200 text-stone-700">
                {fileName}
              </Badge>
            )}
          </div>
        </div>
        {
          error && (
            <Card className="mb-8 border-red-200 bg-red-50">
              <CardContent>
                <p className="text-red-600">
                  Error: {error}
                </p>
              </CardContent>
            </Card>
          )
        }
        {
          vizData && (
            <div className="space-y-8 mt-4">
              <Card>
                <CardHeader>Top Predictions</CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {
                      vizData.predictions.slice(0, 3).map((pred, i) => (
                        <div key={pred.class} className="space-y-2">
                          <div className="flex items-center justify-between">
                            <div className="text-lg font-medium text-stone-700">
                              {getEmojiForClass(pred.class)} {" "}
                              <span>{pred.class.replaceAll("_", " ")}</span>
                            </div>
                            <Badge
                              variant={i == 0 ? "default" : "secondary"}>
                              {(pred.confidence * 100).toFixed(1)}%
                            </Badge>
                          </div>
                          <Progress
                            value={pred.confidence * 100}
                            className="h-2"
                          />
                        </div>
                      ))
                    }

                  </div>
                </CardContent>
              </Card>
            </div>
          )
        }
      </div>
    </main>
  );
}

