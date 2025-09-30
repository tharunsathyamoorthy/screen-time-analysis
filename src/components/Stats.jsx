import React from 'react';

function Stats({ data }) {
  const avgScreenTime = (
    data.reduce((sum, row) => sum + Number(row['ScreenTimeHours'] || 0), 0) / data.length
  ).toFixed(2);

  return (
    <section>
      <h2>Summary Stats</h2>
      <ul>
        <li>Average Screen Time: {avgScreenTime} hours/day</li>
        <li>Total Participants: {data.length}</li>
      </ul>
    </section>
  );
}

export default Stats;
