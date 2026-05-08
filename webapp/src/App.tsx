import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import Upload from './pages/Upload'
import Transform from './pages/Transform'
import Model from './pages/Model'
import Results from './pages/Results'
import Tune from './pages/Tune'
import Visualize from './pages/Visualize'
import Optimize from './pages/Optimize'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/upload" replace />} />
          <Route path="upload" element={<Upload />} />
          <Route path="transform" element={<Transform />} />
          <Route path="model" element={<Model />} />
          <Route path="results" element={<Results />} />
          <Route path="tune" element={<Tune />} />
          <Route path="visualize" element={<Visualize />} />
          <Route path="optimize" element={<Optimize />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
