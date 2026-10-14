import '@testing-library/jest-dom/vitest';
import { render, screen, within } from '@testing-library/react';
import { MemoryRouter, matchRoutes } from 'react-router-dom';
import { describe, expect, it } from 'vitest';
import component1Routes from '../routes';
import UploadPage from '../pages/UploadPage';

describe('Production inspection navigation', () => {
  it('registers only Function 1 routes', () => {
    expect(component1Routes.map(route => route.path)).toEqual([
      '/component1', '/component1/history', '/component1/inspections/:inspectionId',
    ]);
    for (const path of ['/component1/evaluations', '/component1/evaluations/old-run', '/component1/evaluations/old-run/samples']) {
      expect(matchRoutes(component1Routes, path)).toBeNull();
    }
  });
  it('retains inspection navigation without a research workspace', () => {
    render(<MemoryRouter><UploadPage /></MemoryRouter>);
    const navigation = within(screen.getByRole('navigation', { name:'Inspection navigation' }));
    expect(navigation.getAllByRole('link').map(link => link.textContent)).toEqual(['New Inspection','Inspection History']);
    expect(screen.queryByText(/scientific evaluation|separate evaluation workspace|^research$/i)).not.toBeInTheDocument();
    expect(screen.getByRole('button',{name:'Run Inspection'})).toBeInTheDocument();
  });
});
