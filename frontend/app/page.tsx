// frontend/app/page.tsx

import React, { useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { BarChart, PieChart } from "@/components/Charts";
import axios from "axios";

export default function Home() {
  const [age, setAge] = useState(30);
  const [bmi, setBmi] = useState(22);
  const [result, setResult] = useState("");

  const handlePredict = async () => {
    const payload = {
      Age: age,
      BMI: bmi,
      Blood_Pressure_Systolic: 120,
      Blood_Pressure_Diastolic: 80,
      Cholesterol: 180,
      Blood_Sugar: 90,
      Diet_Quality: 6,
      Region: "Urban",
      Gender: "Male",
      Physical_Activity: "Moderate"
    };
    try {
      const res = await axios.post("http://localhost:8000/predict/single", payload);
      setResult(res.data.prediction);
    } catch (err) {
      setResult("Error fetching prediction");
    }
  };

  return (
    <main className="p-6 max-w-5xl mx-auto">
      <h1 className="text-3xl font-bold mb-4">Obesity Risk Classifier</h1>

      <Tabs defaultValue="predict" className="mt-4">
        <TabsList>
          <TabsTrigger value="predict">Predict</TabsTrigger>
          <TabsTrigger value="visualize">Visualize</TabsTrigger>
          <TabsTrigger value="retrain">Retrain</TabsTrigger>
        </TabsList>

        <TabsContent value="predict">
          <Card className="mt-4">
            <CardContent className="grid grid-cols-2 gap-4 pt-4">
              <Input type="number" value={age} onChange={e => setAge(+e.target.value)} placeholder="Age" />
              <Input type="number" value={bmi} onChange={e => setBmi(+e.target.value)} placeholder="BMI" />
              <Button className="col-span-2" onClick={handlePredict}>Submit</Button>
              {result && <p className="col-span-2 font-semibold">Risk Level: {result}</p>}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="visualize">
          <div className="grid grid-cols-2 gap-6 mt-4">
            <Card><CardContent><PieChart /></CardContent></Card>
            <Card><CardContent><BarChart /></CardContent></Card>
          </div>
        </TabsContent>

        <TabsContent value="retrain">
          <Card className="mt-4">
            <CardContent className="space-y-4 pt-4">
              <Input type="file" />
              <Button>Trigger Retraining</Button>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </main>
  );
}